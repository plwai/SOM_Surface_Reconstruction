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

#include "SOMwidget.h"

#include "../common/dataset.h"

#include "../../cuda/reduction/reduction.h"
#include "../../cuda/som/som_kernels.h"
#include "../../common/Utilities.h"

#include <QMessageBox>
#include <ctime>
#include <cmath>

#define GPUMLIB_SOM_INITIAL_LEARNING_RATE (cudafloat(0.5))
#define WEIGHTS_OUTPUT_CPU "weights_cpu.txt"
#define WEIGHTS_OUTPUT_GPU "weights_gpu.txt"

#define MAP_OUTPUT_CPU "map_cpu.txt"
#define MAP_OUTPUT_GPU "map_gpu.txt"

namespace GPUMLib {

	void SOMwidget::LogConfiguration(LogHTML & log, ParameterValues & parameterValues) {
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
		log.AddColumn("Number of features");
		log.AddColumn(parameterValues["features"]);
		log.EndRow();

		log.EndTable();
	}

	void SOMwidget::Run(ParameterValues & parameterValues, LogHTML & summaryLog, LogHTML & log) {
		bool hasHeader = parameterValues.GetBoolParameter("header");

		int trainSamples = parameterValues.GetIntParameter("trainsamples");
		QString trainfile = parameterValues["trainfile"];

		int features = parameterValues.GetIntParameter("features");

		int maxIterations = parameterValues.GetIntParameter("iterations");

		ProgressInfo progress(this, "SOM - Training network", 0, maxIterations);
		progress.Update("Loading datasets");

		std::unique_ptr<Dataset> dsTrain;

		try {
			dsTrain = std::move(std::unique_ptr<Dataset>(new Dataset(trainfile, hasHeader, false, features, 1, trainSamples, log)));
		} catch (QString & error) {
			QMessageBox(QMessageBox::Warning, "Warning", QString("Error loading the training dataset. ") + error).exec();
			return;
		} catch (...) {
			QMessageBox(QMessageBox::Warning, "Warning", QString("Error loading the training dataset: <i>%1</i>.").arg(trainfile)).exec();
			return;
		}

		int vectors = dsTrain->NumberOfSamples();
		int mapx = parameterValues.GetIntParameter("mapx");
		int mapy = parameterValues.GetIntParameter("mapy");
		int mapz = 10; // temporary

		CudaMatrix<cudafloat> inputs(dsTrain->GetInputs());

		CudaArray<int> targets(vectors);
		for (int i = 0; i < vectors; i++) targets[i] = (int)dsTrain->GetTargets()(i, 0);
		if (DeviceIsGPU()) targets.UpdateDevice();

		CudaMatrix3D<Features> weights(mapz, mapx, mapy);
		InitWeights(weights);

		CudaMatrix3D<int> mapView(mapz, mapy, mapx);
		for (int z = 0; z < mapz; z++) {
			for (int y = 0; y < mapy; y++) {
				for (int x = 0; x < mapx; x++) {
					mapView(z, y, x) = 0;
				}
			}
		}

		CudaArray<int> winNode(vectors);
		for (int i = 0; i < vectors; i++) winNode[i] = 0;

		if (DeviceIsGPU()) {
			mapView.UpdateDevice();
			winNode.UpdateDevice();
		}

		cudafloat mapRadius = std::max(mapx, mapy) / cudafloat(2.0);
		cudafloat timeConstant = maxIterations / std::log(mapRadius);

		progress.Update("Training network ...");

		if (DeviceIsCPU()) {
			clock_t initialTime = clock();
			int iteration = TrainCPU(progress, maxIterations, inputs, targets, weights, mapView, winNode, mapRadius, timeConstant, summaryLog, log);
			double elapsedTime = (clock() - initialTime) / 1000.0;

			summaryLog.AppendParagraph(QString("Training complete (%1 iterations).").arg(iteration));
			log.AppendParagraph(QString("CPU Training time (%1 iterations) : %2s").arg(iteration).arg(elapsedTime));

			log.AppendLine();
			log.AppendLine("Map:");

			ShowMapView(log, mapView, MAP_OUTPUT_CPU);

			WriteWeights(weights, WEIGHTS_OUTPUT_CPU);
		} else {
			/*clock_t initialTime = clock();
			int iteration = TrainGPU(progress, maxIterations, inputs, targets, weights, mapView, winNode, mapRadius, timeConstant, summaryLog, log);
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
			}*/
		}

		progress.End();

		summaryLog.Append(log.ToString());
	}

	int SOMwidget::TrainCPU(ProgressInfo & progress, int iteration, CudaMatrix<cudafloat> & inputData, CudaArray<int> & targets, CudaMatrix3D<Features> & weights, CudaMatrix3D<int> & mapView, CudaArray<int> & winNode, cudafloat mapRadius, cudafloat timeConstant, LogHTML & summaryLog, LogHTML & log) {
		cudafloat learningRate = GPUMLIB_SOM_INITIAL_LEARNING_RATE;

		int features = (int)inputData.Columns();
		int samples = (int)inputData.Rows();
		int mapx = (int)mapView.DimZ();
		int mapy = (int)mapView.DimY();
		int mapz = (int)mapView.DimX();

		int currentIteration = 0;

		for (int iter = iteration; iter > 0; --iter) {
			currentIteration++;

			cudafloat neighbourhoodRadius = mapRadius * exp(cudafloat(-currentIteration) / timeConstant);
			cudafloat squareNeighbourhoodRadius = neighbourhoodRadius * neighbourhoodRadius;

			for (int vector = 0; vector < samples; vector++) {
				FindBestMatchingUnit(vector, inputData, targets, weights, mapView, winNode);

				for (int y = 0; y < mapy; y++) {
					for (int x = 0; x < mapx; x++) {
						for (int z = 0; z < mapz; z++) {
							int win = winNode[vector];
							int winx = win % mapx;
							int winy = (win % (mapx * mapy)) / mapx;
							int winz = (win / (mapx * mapy));

							cudafloat dx = winx - x;
							cudafloat dy = winy - y;
							cudafloat dz = winz - z;

							cudafloat distanceFromNode = dx * dx + dy * dy + dz * dz;

							//if (distanceFromNode < squareNeighbourhoodRadius) {
							if (distanceFromNode < squareNeighbourhoodRadius) {
								//cudafloat m_dInfluence = exp(-(distanceFromNode) / (2 * squareNeighbourhoodRadius));
								cudafloat m_dInfluence = exp(-(distanceFromNode) / (2 * squareNeighbourhoodRadius));

								weights(z, x, y).x += (cudafloat)(learningRate * m_dInfluence * (inputData(vector, 0) - weights(z, x, y).x));
								weights(z, x, y).y += (cudafloat)(learningRate * m_dInfluence * (inputData(vector, 1) - weights(z, x, y).y));
								weights(z, x, y).z += (cudafloat)(learningRate * m_dInfluence * (inputData(vector, 2) - weights(z, x, y).z));
								
							}
						}
					}
				}

				learningRate = (cudafloat)(GPUMLIB_SOM_INITIAL_LEARNING_RATE * exp(cudafloat(-currentIteration) / iter));

				NormalizeWeights(weights);
			}

			if (progress.WasCanceled()) break;

			if (progress.NeedsUpdating()) progress.SetValue(currentIteration);
		}

		return currentIteration;
	}

	QString CudaError(int iteration, cudaError_t error) {
		return QString("A CUDA <b>error</b> has occurred during training (iteration %1): %2").arg(iteration).arg(cudaGetErrorString(error));
	}

	/*int SOMwidget::TrainGPU(ProgressInfo & progress, int iterations, CudaMatrix<cudafloat> & inputData, CudaArray<int> & targets, CudaMatrix3D<cudafloat> & weights, CudaMatrix<int> & mapView, CudaArray<int> & winNode, cudafloat mapRadius, cudafloat timeConstant, LogHTML & summaryLog, LogHTML & log) {
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

				error = UpdateWeightsSOM(blockSizeUpdateWeights, bmu.DevicePointer(), mapView.DevicePointer(), mapx, mapy, inputData.DevicePointer(), vector, features, targets[vector], squareNeighbourhoodRadius, weights.DevicePointer(), learningRate);


				if (error != cudaSuccess) {
					QString e = CudaError(currentIteration, error);
					summaryLog.Append(e);
					log.Append(e);

					return 0;
				}

				error = NormalizeWeightsSOM(gridMap, threadsFeatures, mapx, mapy, features, weights.DevicePointer());

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
	}*/

	void SOMwidget::FindBestMatchingUnit(int vector, CudaMatrix<cudafloat> & inputData, CudaArray<int> & target, CudaMatrix3D<Features> & weights, CudaMatrix3D<int> & mapView, CudaArray<int> & winNode) {
		cudafloat lowestDistance = MAX_CUDAFLOAT;

		int winx = -1;
		int winy = -1;
		int winz = -1;

		int rows = (int)mapView.DimY();
		int columns = (int)mapView.DimZ();
		int depth = (int)mapView.DimX();

		for (int y = 0; y < rows; y++) {
			for (int x = 0; x < columns; x++) {
				for (int z = 0; z < depth; z++) {
					cudafloat distance = CalculateDistance(vector, x, y, z, inputData, weights);

					if (distance < lowestDistance) {
						lowestDistance = distance;
						winx = x;
						winy = y;
						winz = z;
					}
				}
			}
		}

		winNode[vector] = winy * columns + winx + winz * columns * rows;
		mapView(winz, winy, winx) = target[vector];
	}

	cudafloat SOMwidget::CalculateDistance(int input, int wx, int wy, int wz, CudaMatrix<cudafloat> & inputData, CudaMatrix3D<Features> & weights) {
		cudafloat distance = 0.0f;

		int features = (int)inputData.Columns();
	
		cudafloat d = inputData(input, 0) - weights(wz, wx, wy).x;
		distance += d * d;

		d = inputData(input, 1) - weights(wz, wx, wy).y;
		distance += d * d;

		d = inputData(input, 2) - weights(wz, wx, wy).z;
		distance += d * d;

		return sqrt(distance);
	}

	void SOMwidget::InitWeights(CudaMatrix3D<Features> & weights) {
		for (size_t z = 0; z < weights.DimZ(); z++) { // mapy
			for (size_t y = 0; y < weights.DimY(); y++) { // mapx
				for (size_t x = 0; x < weights.DimX(); x++) { // mapz
					Features feature;
					feature.x = (cudafloat)rand() / RAND_MAX;
					feature.y = (cudafloat)rand() / RAND_MAX;
					feature.z = (cudafloat)rand() / RAND_MAX;
					weights(x, y, z) = feature;
				}
			}
		}

		NormalizeWeights(weights);

		if (DeviceIsGPU()) weights.UpdateDevice();
	}

	void SOMwidget::NormalizeWeights(CudaMatrix3D<Features> & weights) {
		for (size_t z = 0; z < weights.DimZ(); z++) { // mapy
			for (size_t y = 0; y < weights.DimY(); y++) { // mapx
				for (size_t x = 0; x < weights.DimX(); x++) { // mapz
					double norm = 0.0;
					Features current_weight;

					current_weight = weights(x, y, z);
					norm += current_weight.x * current_weight.x;
					norm += current_weight.y * current_weight.y;
					norm += current_weight.z * current_weight.z;

					norm = 1.0 / (sqrt(norm));

					current_weight.x *= (cudafloat)norm;
					current_weight.y *= (cudafloat)norm;
					current_weight.z *= (cudafloat)norm;

					weights(x, y, z) = current_weight;

				}
			}
		}
	}

	void SOMwidget::WriteWeights(CudaMatrix3D<Features> & weights, char * weightsOutput) {
		FILE *fw = fopen(weightsOutput, "w");

		for (size_t z = 0; z < weights.DimZ(); z++) { // mapy
			for (size_t y = 0; y < weights.DimY(); y++) { // mapx
				for (size_t x = 0; x < weights.DimX(); x++) { // mapz
					fprintf(fw, "%.4lf ", weights(x, y, z).x);
					fprintf(fw, "%.4lf ", weights(x, y, z).y);
					fprintf(fw, "%.4lf ", weights(x, y, z).z);
					fprintf(fw, "\n");
				}
			}
		}

		fclose(fw);
	}

	void SOMwidget::ShowMapView(LogHTML & log, CudaMatrix3D<int> & mapView, char * mapOutput) {
		FILE *fs = fopen(mapOutput, "w");

		for (size_t k = 0; k < mapView.DimX(); k++) {
			log.BeginTable(0);

			for (size_t i = 0; i < mapView.DimY(); i++) {
				log.BeginRow();

				for (size_t j = 0; j < mapView.DimZ(); j++) {
					log.AddColumn(mapView(k, i, j));
					fprintf(fs, "%d ", mapView(k, i, j));
				}
				log.EndRow();
				fprintf(fs, "\n");
			}

			log.EndTable();
		}
	}

} // namespace GPUMLib
