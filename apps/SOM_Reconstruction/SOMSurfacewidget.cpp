/*
	Noel Lopes is a Professor at the Polytechnic of Guarda, Portugal
	and a Researcher at the CISUC - University of Coimbra, Portugal
	Copyright (C) 2009-2015 Noel de Jesus Mendonça Lopes

	Siti Mariyam Shamsuddin is a Professor at the Faculty of Computing
	Universiti Teknologi Malaysia (UTM), Malaysia and researcher at
	UTM Big Data Centre, Malaysia

	Shafaatunnur Hasan is a full time researcher at UTM Big Data Centre,
	Malaysia

	Copyright (C) 2009-2015 Noel de Jesus Mendonça Lopes
	                        Siti Mariyam Shamsuddin
	                        Shafaatunnur Hasan

	This file is part of GPUMLib.

	GPUMLib is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program. If not, see <http://www.gnu.org/licenses/>.
	*/

#include "SOMSurfacewidget.h"

#include "../common/dataset3D.h"

#include "../../cuda/reduction/reduction.h"
#include "../../cuda/som_surface/som_surface_kernels.h"
#include "../../common/Utilities.h"

#include <QMessageBox>
#include <ctime>
#include <cmath>
#include <string>

#ifdef PCL_EXIST

#include "SurfaceViewer.h"

#endif

#define GPUMLIB_SOM_INITIAL_LEARNING_RATE (cudafloat(0.5))
#define WEIGHTS_OUTPUT_CPU "weights_cpu.txt"
#define WEIGHTS_OUTPUT_GPU "weights_gpu.txt"

#define MAP_OUTPUT_CPU "map_cpu.txt"
#define MAP_OUTPUT_GPU "map_gpu.txt"

#define PLY_OUTPUT_CPU "3D_cpu.ply"
#define PLY_OUTPUT_GPU "3D_gpu.ply"

#define PROCESSED_FILE "processed_file.txt"

namespace GPUMLib {

	void SOMSurfacewidget::LogConfiguration(LogHTML & log, ParameterValues & parameterValues) {
		log.AppendSection("SOM configuration");
		log.BeginTable(0, 1);

		log.BeginRow();
		log.AddColumn("Map width (X dimension)");
		log.AddColumn(parameterValues["mapx"]);
		log.EndRow();

		log.BeginRow();
		log.AddColumn("Map height (Y dimension)");
		log.AddColumn(parameterValues["mapy"]);
		log.EndRow();

		log.BeginRow();
		log.AddColumn("Iterations");
		log.AddColumn(parameterValues["iterations"]);
		log.EndRow();

		log.BeginRow();
		log.AddColumn("The datasets contain an header line");
		log.AddColumn(parameterValues["header"]);
		log.EndRow();

		log.BeginRow();
		log.AddColumn("Training filename");
		log.AddColumn(parameterValues["trainfile"]);
		log.EndRow();

		if (parameterValues.GetIntParameter("trainsamples") > 0) {
			log.BeginRow();
			log.AddColumn("Training samples");
			log.AddColumn(parameterValues["trainsamples"]);
			log.EndRow();
		}

		log.BeginRow();
		log.AddColumn("Map Type");
		log.AddColumn(parameterValues["maptype"]);
		log.EndRow();

		if (parameterValues.GetIntParameter("multi") > 0) {
			log.BeginRow();
			log.AddColumn("Number of Layer");
			log.AddColumn(parameterValues["multi"]);
			log.EndRow();

			log.BeginRow();
			log.AddColumn("Map order");
			if (parameterValues.GetIntParameter("layerorder") == 1) {
				log.AddColumn("Increasing");
			}
			else {
				log.AddColumn("Decreasing");
			}
			
			log.EndRow();

			log.BeginRow();
			log.AddColumn("Map Interval");
			log.AddColumn(parameterValues["mapinter"]);
			log.EndRow();
		}

		log.EndTable();
	}

	void SOMSurfacewidget::Run(ParameterValues & parameterValues, LogHTML & summaryLog, LogHTML & log) {
		bool hasHeader = parameterValues.GetBoolParameter("header");

		int trainSamples = parameterValues.GetIntParameter("trainsamples");
		QString trainfile = parameterValues["trainfile"];

		int features = 3; // x y z coordinate point

		int maxIterations = parameterValues.GetIntParameter("iterations");


		int layerNum = parameterValues.GetIntParameter("multi");

		int layerOrder = parameterValues.GetIntParameter("layerorder");

		int mapInter = parameterValues.GetIntParameter("mapinter");

		ProgressInfo progress(this, "SOM - Training network", 0, maxIterations);
		progress.Update("Loading datasets");

		std::unique_ptr<Dataset3D> dsTrain;

		const int RESCALE_MIN = 0;
		const int RESCALE_MAX = 1024;

		int mapx = parameterValues.GetIntParameter("mapx");
		int mapz = parameterValues.GetIntParameter("maptype");
		int mapy = parameterValues.GetIntParameter("mapy") / mapz;

		for (int layer = 0; layer < layerNum; layer++) {
			if (layer) {
				if (layerOrder == 1) {
					mapx += mapInter;
					mapy += mapInter;
				}
				else {
					mapx -= mapInter;
					mapy -= mapInter;
				}

				if (DeviceIsGPU()) {
					trainfile = PLY_OUTPUT_GPU;
				}
				else {
					trainfile = PLY_OUTPUT_CPU;
				}

				maxIterations += 500;
				ProgressInfo progress(this, "SOM - Training network", 0, maxIterations);

				if (parameterValues.GetIntParameter("random") == 0) {
					srand(time(NULL));
				}
			}

			try {
				dsTrain = std::move(std::unique_ptr<Dataset3D>(new Dataset3D(trainfile, hasHeader, RESCALE_MIN, RESCALE_MAX, features, 1, trainSamples, log)));
			} catch (QString & error) {
				QMessageBox(QMessageBox::Warning, "Warning", QString("Error loading the training dataset. ") + error).exec();
				return;
			} catch (...) {
				QMessageBox(QMessageBox::Warning, "Warning", QString("Error loading the training dataset: <i>%1</i>.").arg(trainfile)).exec();
				return;
			}

			int vectors = dsTrain->NumberOfSamples();

			CudaMatrix<cudafloat> inputs(dsTrain->GetInputs());

			WriteProcessedInputFile(inputs, vectors, PROCESSED_FILE);

			CudaArray<int> targets(vectors);
			for (int i = 0; i < vectors; i++) targets[i] = (int)dsTrain->GetTargets()(i, 0);
			if (DeviceIsGPU()) targets.UpdateDevice();

			CudaMatrix3D<cudafloat> weights(features, mapx, mapy * mapz);
			InitWeights(weights, RESCALE_MAX);

			CudaMatrix<int> mapView(mapy * mapz, mapx);
			for (int y = 0; y < mapy * mapz; y++) {
				for (int x = 0; x < mapx; x++) {
					mapView(y, x) = 0;
				}
			}

			CudaArray<int> winNode(vectors);
			for (int i = 0; i < vectors; i++) winNode[i] = 0;

			if (DeviceIsGPU()) {
				mapView.UpdateDevice();
				winNode.UpdateDevice();
			}

			cudafloat mapRadius = std::max(std::max(mapx, mapy / mapz), mapz) / cudafloat(2.0);
			cudafloat timeConstant = maxIterations / std::log(mapRadius);

			progress.Update(("Training network ... layer " + std::to_string(layer + 1)).c_str());

			if (DeviceIsCPU()) {
				clock_t initialTime = clock();
				int iteration = TrainCPU(progress, maxIterations, inputs, targets, weights, mapView, winNode, mapRadius, timeConstant, summaryLog, log, mapz);
				double elapsedTime = (clock() - initialTime) / 1000.0;

				summaryLog.AppendParagraph(QString("Training complete (%1 iterations).").arg(iteration));
				log.AppendParagraph(QString("CPU Training time (%1 iterations) : %2s").arg(iteration).arg(elapsedTime));

				log.AppendLine();
				log.AppendLine("Map:");

				ShowMapView(log, mapView, MAP_OUTPUT_CPU);

				WriteWeights(weights, WEIGHTS_OUTPUT_CPU);

				if (layerNum != 1 && layer != layerNum - 1) {
					WriteLayerPLYFile(inputs, mapView, mapz, PLY_OUTPUT_CPU);
				}
				else {
					WritePLYFile(weights, mapView, mapz, PLY_OUTPUT_CPU);
				}
			}
			else {
				clock_t initialTime = clock();
				int iteration = TrainGPU(progress, maxIterations, inputs, targets, weights, mapView, winNode, mapRadius, timeConstant, summaryLog, log, mapz);
				cudaThreadSynchronize();
				double elapsedTime = (clock() - initialTime) / 1000.0;

				if (iteration > 0) {
					summaryLog.AppendParagraph(QString("Training complete (%1 iterations).").arg(iteration));
					log.AppendParagraph(QString("GPU Training time (%1 iterations) : %2s").arg(iteration).arg(elapsedTime));

					log.AppendLine();
					log.AppendLine("Map:");

					mapView.UpdateHost();
					ShowMapView(log, mapView, MAP_OUTPUT_GPU);

					weights.UpdateHost();
					WriteWeights(weights, WEIGHTS_OUTPUT_GPU);

					if (layerNum != 1 && layer != layerNum - 1) {
						WriteLayerPLYFile(inputs, mapView, mapz, PLY_OUTPUT_GPU);
					}
					else {
						WritePLYFile(weights, mapView, mapz, PLY_OUTPUT_GPU);
					}
				}
			}
		}

		progress.End();

		summaryLog.Append(log.ToString());

	#ifdef PCL_EXIST

		SurfaceViewer sv;
		sv.initViewer(RESCALE_MAX);
		sv.loadMesh(DeviceIsCPU() ? PLY_OUTPUT_CPU : PLY_OUTPUT_GPU);
		sv.run();
		
	#endif
	}

	int SOMSurfacewidget::TrainCPU(ProgressInfo & progress, int iteration, CudaMatrix<cudafloat> & inputData, CudaArray<int> & targets, CudaMatrix3D<cudafloat> & weights, CudaMatrix<int> & mapView, CudaArray<int> & winNode, cudafloat mapRadius, cudafloat timeConstant, LogHTML & summaryLog, LogHTML & log, int & mapType) {
		cudafloat learningRate = GPUMLIB_SOM_INITIAL_LEARNING_RATE;

		int features = (int)inputData.Columns();
		int samples = (int)inputData.Rows();
		int mapx = (int)mapView.Columns();
		int mapy = (int)mapView.Rows();

		int currentIteration = 0;

		for (int iter = iteration; iter > 0; --iter) {
			currentIteration++;

			cudafloat neighbourhoodRadius = mapRadius * exp(cudafloat(-currentIteration) / timeConstant);
			cudafloat squareNeighbourhoodRadius = neighbourhoodRadius * neighbourhoodRadius;

			for (int vector = 0; vector < samples; vector++) {
				FindBestMatchingUnit(vector, inputData, targets, weights, mapView, winNode);

				int win = winNode[vector];
				int winx = win % mapx;
				int winy = win / mapx;

				if (mapType == 1) {
					// Basic Map
					for (int y = 0; y < mapy; y++) {
						for (int x = 0; x < mapx; x++) {


							cudafloat dx = winx - x;
							cudafloat dy = winy - y;

							cudafloat distanceFromNode = dx * dx + dy * dy;

							if (distanceFromNode < squareNeighbourhoodRadius) {
								cudafloat m_dInfluence = exp(-(distanceFromNode) / (2 * squareNeighbourhoodRadius));

								for (int k = 0; k < features; k++) {
									weights(k, x, y) += (cudafloat)(learningRate * m_dInfluence * (inputData(vector, k) - weights(k, x, y)));
								}
							}
						}
					}
				}
				else {
					// Dual Layer Map
					for (int y = 0; y < mapy; y++) {
						int tempY = y >= (mapy / 2) ? y - (mapy / 2) : y;
						int tempwinY = winy >= (mapy / 2) ? winy - (mapy / 2) : winy;

						cudafloat dz = abs((int)(winy / (mapy / 2)) - (int)(y / (mapy / 2)));

						cudafloat dy = 0;
						cudafloat leftDist = 0;
						cudafloat rightDist = 0;
						cudafloat topDist = 0;
						cudafloat botDist = 0;

						if (dz != 0) {
							// 4 direction distance
							// left
							leftDist = (tempwinY - tempY) * (tempwinY - tempY);

							// right
							rightDist = leftDist;

							// top
							topDist = ((tempY + tempwinY + 1) * (tempY + tempwinY + 1));

							// bottom
							botDist = ((((mapy / 2) - tempY) + ((mapy / 2) - tempwinY - 1)) * (((mapy / 2) - tempY) + ((mapy / 2) - tempwinY - 1)));
						}
						else {
							dy = winy - y;
						}

						for (int x = 0; x < mapx; x++) {
							cudafloat dx = 0;
							cudafloat distanceFromNode = 0;

							if (dz != 0) {
								// 4 direction distance
								// left
								leftDist += ((x + winx + 1) * (x + winx + 1));

								// right
								rightDist += (((mapx - x) + (mapx - winx - 1)) * ((mapx - x) + (mapx - winx - 1)));

								// top
								topDist += ((winx - x) * (winx - x));

								// bottom
								botDist += ((winx - x) * (winx - x));

							}
							else {
								dx = winx - x;
							}

							if (dz) {
								distanceFromNode = leftDist < rightDist ? leftDist : rightDist;
								distanceFromNode = distanceFromNode < topDist ? distanceFromNode : topDist;
								distanceFromNode = distanceFromNode < botDist ? distanceFromNode : botDist;
							}
							else {
								distanceFromNode = dx * dx + dy * dy;
							}

							if (distanceFromNode < squareNeighbourhoodRadius) {
								cudafloat m_dInfluence = exp(-(distanceFromNode) / (2 * squareNeighbourhoodRadius));

								for (int k = 0; k < features; k++) {
									weights(k, x, y) += (cudafloat)(learningRate * m_dInfluence * (inputData(vector, k) - weights(k, x, y)));
								}
							}
						}
					}
				}

				learningRate = (cudafloat)(GPUMLIB_SOM_INITIAL_LEARNING_RATE * exp(cudafloat(-currentIteration) / iter));

			}

			if (progress.WasCanceled()) break;

			if (progress.NeedsUpdating()) progress.SetValue(currentIteration);
		}

		return currentIteration;
	}

	QString CudaError(int iteration, cudaError_t error) {
		return QString("A CUDA <b>error</b> has occurred during training (iteration %1): %2").arg(iteration).arg(cudaGetErrorString(error));
	}

	int SOMSurfacewidget::TrainGPU(ProgressInfo & progress, int iterations, CudaMatrix<cudafloat> & inputData, CudaArray<int> & targets, CudaMatrix3D<cudafloat> & weights, CudaMatrix<int> & mapView, CudaArray<int> & winNode, cudafloat mapRadius, cudafloat timeConstant, LogHTML & summaryLog, LogHTML & log, int & mapType) {
		cudafloat learningRate = GPUMLIB_SOM_INITIAL_LEARNING_RATE;

		int features = (int)inputData.Columns();
		int samples = (int)inputData.Rows();
		int mapx = (int)mapView.Columns();
		int mapy = (int)mapView.Rows();

		DeviceMatrix<cudafloat> distances(mapy, mapx);
		CudaArray<int> bmu(1);

		int threadsFeatures = NumberThreadsPerBlockThatBestFit(features);

		dim3 gridMap(mapx, mapy);

		dim3 blockSizeUpdateWeights(features, mapx, mapy);
		MakeSureBlockDoesNotHaveTooMuchThreads(blockSizeUpdateWeights);

		int currentIteration = 0;

		for (int iter = iterations; iter > 0; --iter) {
			currentIteration++;

			cudafloat neighbourhoodRadius = mapRadius * exp(cudafloat(-currentIteration) / timeConstant);
			cudafloat squareNeighbourhoodRadius = neighbourhoodRadius * neighbourhoodRadius;

			for (int vector = 0; vector < samples; vector++) {
				cudaError_t error = ComputeDistancesSOM(gridMap, threadsFeatures, inputData.DevicePointer(), weights.DevicePointer(), vector, features, distances.Pointer());

				if (error != cudaSuccess) {
					QString e = CudaError(currentIteration, error);
					summaryLog.Append(e);
					log.Append(e);

					return 0;
				}

				// Find the best matching unit
				Reduction::MinIndex(distances, bmu.GetDeviceArray());

				error = cudaGetLastError();
				if (error != cudaSuccess) {
					QString e = CudaError(currentIteration, error);
					summaryLog.Append(e);
					log.Append(e);

					return 0;
				}

				if (mapType == 1) {
					error = UpdateWeightsSOM(blockSizeUpdateWeights, bmu.DevicePointer(), mapView.DevicePointer(), mapx, mapy, inputData.DevicePointer(), vector, features, targets[vector], squareNeighbourhoodRadius, weights.DevicePointer(), learningRate);
				}
				else {
					error = UpdateWeightsSOMDual(blockSizeUpdateWeights, bmu.DevicePointer(), mapView.DevicePointer(), mapx, mapy, inputData.DevicePointer(), vector, features, targets[vector], squareNeighbourhoodRadius, weights.DevicePointer(), learningRate);
				}


				if (error != cudaSuccess) {
					QString e = CudaError(currentIteration, error);
					summaryLog.Append(e);
					log.Append(e);

					return 0;
				}

				learningRate = (cudafloat)(GPUMLIB_SOM_INITIAL_LEARNING_RATE * exp(cudafloat(-currentIteration) / iter));
			}

			if (progress.WasCanceled()) break;

			if (progress.NeedsUpdating()) progress.SetValue(currentIteration);
		}

		return currentIteration;
	}

	void SOMSurfacewidget::FindBestMatchingUnit(int vector, CudaMatrix<cudafloat> & inputData, CudaArray<int> & target, CudaMatrix3D<cudafloat> & weights, CudaMatrix<int> & mapView, CudaArray<int> & winNode) {
		cudafloat lowestDistance = MAX_CUDAFLOAT;

		int winx = -1;
		int winy = -1;

		int rows = (int)mapView.Rows();
		int columns = (int)mapView.Columns();

		for (int y = 0; y < rows; y++) {
			for (int x = 0; x < columns; x++) {
				cudafloat distance = CalculateDistance(vector, x, y, inputData, weights);

				if (distance < lowestDistance) {
					lowestDistance = distance;
					winx = x;
					winy = y;
				}
			}
		}

		winNode[vector] = winy * columns + winx;
		mapView(winy, winx) = target[vector];
	}

	cudafloat SOMSurfacewidget::CalculateDistance(int input, int wx, int wy, CudaMatrix<cudafloat> & inputData, CudaMatrix3D<cudafloat> & weights) {
		cudafloat distance = 0.0f;

		int features = (int)inputData.Columns();
		for (int f = 0; f < features; f++) {
			cudafloat d = inputData(input, f) - weights(f, wx, wy);
			distance += d * d;
		}

		return sqrt(distance);
	}

	void SOMSurfacewidget::InitWeights(CudaMatrix3D<cudafloat> & weights, int maxScale) {
		for (size_t z = 0; z < weights.DimZ(); z++) { // mapy
			for (size_t y = 0; y < weights.DimY(); y++) { // mapx
				for (size_t x = 0; x < weights.DimX(); x++) { // features
					weights(x, y, z) = (cudafloat)(rand() % maxScale);;
				}
			}
		}

		if (DeviceIsGPU()) weights.UpdateDevice();
	}

	void SOMSurfacewidget::WriteWeights(CudaMatrix3D<cudafloat> & weights, char * weightsOutput) {
		FILE *fw = fopen(weightsOutput, "w");

		for (size_t z = 0; z < weights.DimZ(); z++) { // mapy
			for (size_t y = 0; y < weights.DimY(); y++) { // mapx
				for (size_t x = 0; x < weights.DimX(); x++) { // features
					fprintf(fw, "%.4lf ", weights(x, y, z));
				}
				fprintf(fw, "\n");
			}
		}

		fclose(fw);
	}

	void SOMSurfacewidget::ShowMapView(LogHTML & log, CudaMatrix<int> & mapView, char * mapOutput) {
		FILE *fs = fopen(mapOutput, "w");

		log.BeginTable(0);

		for (size_t i = 0; i < mapView.Rows(); i++) {
			log.BeginRow();

			for (size_t j = 0; j < mapView.Columns(); j++) {
				log.AddColumn(mapView(i, j));
				fprintf(fs, "%d ", mapView(i, j));
			}
			log.EndRow();
			fprintf(fs, "\n");
		}

		log.EndTable();

		fclose(fs);
	}

	void SOMSurfacewidget::WritePLYFile(CudaMatrix3D<cudafloat> & weights, CudaMatrix<int> & mapView, int mapz, char * plyOutput) {
		FILE *fs = fopen(plyOutput, "w");

		std::string plyTemplate = "";
		int vertexNum, faceNum;

		int mapx = (int)mapView.Columns();
		int mapy = (int)mapView.Rows();

		// Prepare header template
		vertexNum = mapy * mapx;

		if (mapz == 1) {
			// Basic 2D Map
			faceNum = (mapy - 1) * (mapy - 1) * 2;
		}
		else {
			// Dual Layer Map
			faceNum = (4 * ((mapy / 2) - 1)) + (mapx - 1) * 2 * mapy;
		}

		plyTemplate += "ply\nformat ascii 1.0\nelement vertex ";
		plyTemplate += std::to_string(vertexNum);
		plyTemplate += "\nproperty float x\nproperty float y\nproperty float z\nelement face ";
		plyTemplate += std::to_string(faceNum);
		plyTemplate += "\nproperty list uchar int vertex_indices\nend_header\n";

		// Write header
		fprintf(fs, plyTemplate.c_str());

		// Write points
		for (size_t z = 0; z < weights.DimZ(); z++) { // mapy
			for (size_t y = 0; y < weights.DimY(); y++) { // mapx
				for (size_t x = 0; x < weights.DimX(); x++) { // features
					fprintf(fs, "%.4lf ", weights(x, y, z));
				}
				fprintf(fs, "\n");
			}
		}

		// Write surface connection
		if (mapz == 1) {
			// Basic 2D Map

			// Center Connection
			for (int i = 0; i < mapy - 1; i++) {
				for (int j = 0; j < mapx - 1; j++) {
					fprintf(fs, "3 %d %d %d\n", getMapLocation(i, j, mapx), getMapLocation(i, j + 1, mapx), getMapLocation(i + 1, j + 1, mapx));
					fprintf(fs, "3 %d %d %d\n", getMapLocation(i, j, mapx), getMapLocation(i + 1, j + 1, mapx), getMapLocation(i + 1, j, mapx));
				}
			}
		}
		else {
			// Dual Layer Map

			// Center Connection Layer 1
			for (int i = 0; i < mapy / 2 - 1; i++) {
				for (int j = 0; j < mapx - 1; j++) {
					fprintf(fs, "3 %d %d %d\n", getMapLocation(i, j + 1, mapx), getMapLocation(i, j, mapx), getMapLocation(i + 1, j + 1, mapx));
					fprintf(fs, "3 %d %d %d\n", getMapLocation(i + 1, j + 1, mapx), getMapLocation(i, j, mapx), getMapLocation(i + 1, j, mapx));
				}
			}

			// Center Connection Layer 2
			for (int i = mapy / 2; i < mapy - 1; i++) {
				for (int j = 0; j < mapx - 1; j++) {
					fprintf(fs, "3 %d %d %d\n", getMapLocation(i, j, mapx), getMapLocation(i, j + 1, mapx), getMapLocation(i + 1, j + 1, mapx));
					fprintf(fs, "3 %d %d %d\n", getMapLocation(i, j, mapx), getMapLocation(i + 1, j + 1, mapx), getMapLocation(i + 1, j, mapx));
				}
			}

			// Sides Connection ( Top and Bottom )
			for (int j = 0; j < mapx - 1; j++) {
				fprintf(fs, "3 %d %d %d\n", getMapLocation(mapy / 2, j + 1, mapx), getMapLocation(mapy / 2, j, mapx), getMapLocation(0, j + 1, mapx));
				fprintf(fs, "3 %d %d %d\n", getMapLocation(0, j, mapx), getMapLocation(0, j + 1, mapx), getMapLocation(mapy / 2, j, mapx));

				fprintf(fs, "3 %d %d %d\n", getMapLocation(mapy / 2 - 1, j + 1, mapx), getMapLocation(mapy / 2 - 1, j, mapx), getMapLocation(mapy - 1, j + 1, mapx));
				fprintf(fs, "3 %d %d %d\n", getMapLocation(mapy - 1, j, mapx), getMapLocation(mapy - 1, j + 1, mapx), getMapLocation(mapy / 2 - 1, j, mapx));
			}

			// Sides Connection ( Left and Right )
			for (int j = 0; j < mapy / 2 - 1; j++) {
				fprintf(fs, "3 %d %d %d\n", getMapLocation(j, 0, mapx), getMapLocation(j + (mapy / 2), 0, mapx), getMapLocation(j + 1, 0, mapx));
				fprintf(fs, "3 %d %d %d\n", getMapLocation(j + (mapy / 2), 0, mapx), getMapLocation(j + (mapy / 2) + 1, 0, mapx), getMapLocation(j + 1, 0, mapx));

				fprintf(fs, "3 %d %d %d\n", getMapLocation(j + (mapy / 2) + 1, mapx - 1, mapx), getMapLocation(j + (mapy / 2), mapx - 1, mapx), getMapLocation(j, mapx - 1, mapx));
				fprintf(fs, "3 %d %d %d\n", getMapLocation(j, mapx - 1, mapx), getMapLocation(j + 1, mapx - 1, mapx), getMapLocation(j + (mapy / 2) + 1, mapx - 1, mapx));
			}
		}

		fclose(fs);
	}

	void SOMSurfacewidget::WriteLayerPLYFile(CudaMatrix<cudafloat> & inputs, CudaMatrix<int> & mapView, int mapz, char * plyOutput) {
		FILE *fs = fopen(plyOutput, "w");

		std::string plyTemplate = "";
		int vertexNum = 0, faceNum;

		int mapx = (int)mapView.Columns();
		int mapy = (int)mapView.Rows();

		for (int y = 0; y < mapy; y++) {
			for (int x = 0; x < mapx; x++) {
				if (mapView(y, x) != 0) {
					vertexNum++;
				}
			}
		}

		// Prepare header template
		if (mapz == 1) {
			// Basic 2D Map
			faceNum = (mapy - 1) * (mapy - 1) * 2;
		}
		else {
			// Dual Layer Map
			faceNum = (4 * ((mapy / 2) - 1)) + (mapx - 1) * 2 * mapy;
		}

		plyTemplate += "ply\nformat ascii 1.0\nelement vertex ";
		plyTemplate += std::to_string(vertexNum);
		plyTemplate += "\nproperty float x\nproperty float y\nproperty float z\nelement face 0";
		plyTemplate += "\nproperty list uchar int vertex_indices\nend_header\n";

		// Write header
		fprintf(fs, plyTemplate.c_str());

		// Write points
		for (int y = 0; y < mapy; y++) {
			for (int x = 0; x < mapx; x++) {
				for (int f = 0; f < 3; f++) {
					if (mapView(y, x) != 0) {
						fprintf(fs, "%.4lf ", inputs(mapView(y, x) - 1, f));
					}
				}
				if (mapView(y, x) != 0) {
					fprintf(fs, "\n");
				}
			}
		}

		fclose(fs);
	}

	void SOMSurfacewidget::WriteProcessedInputFile(CudaMatrix<cudafloat> & inputs, int vectors , char * outputFile) {
		FILE *fs = fopen(outputFile, "w");

		for (int i = 0; i < vectors; i++) {
			for (int f = 0; f < 3; f++) {
				fprintf(fs, "%.4lf ", inputs(i, f));
			}
			fprintf(fs, "%d\n", i + 1);
		}
		

		fclose(fs);
	}

	int SOMSurfacewidget::getMapLocation(int row, int col, int mapx) {
		return row * mapx + col;
	}

} // namespace GPUMLib
