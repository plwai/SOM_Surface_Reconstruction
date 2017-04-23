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

#ifndef GPUMLIB_SOM_WIDGET_H
#define GPUMLIB_SOM_WIDGET_H

#include "../common/widget/AlgorithmWidget.h"
#include "../common/progress/ProgressInfo.h"

#include "../../memory/CudaMatrix3D.h"
#include "../../memory/CudaMatrix.h"
#include "../../memory/CudaArray.h"

namespace GPUMLib {

	class SOMwidget : public AlgorithmWidget {
		Q_OBJECT

	public:
		explicit SOMwidget(const char * parameterFile, int argc, char ** argv, QWidget *parent = 0) : AlgorithmWidget(parameterFile, argc, argv, parent) {}

	private:

		virtual void Run(ParameterValues & parameterValues, LogHTML & summaryLog, LogHTML & log) override;
		virtual void LogConfiguration(LogHTML & log, ParameterValues & parameterValues) override;

		int TrainCPU(ProgressInfo & progress, int iterations, CudaMatrix<cudafloat> & inputData, CudaArray<int> & targets, CudaMatrix3D<cudafloat> & weights, CudaMatrix<int> & mapView, CudaArray<int> & winNode, cudafloat mapRadius, cudafloat timeConstant, LogHTML & summaryLog, LogHTML & log, int & tools, int & maptype);
		int TrainGPU(ProgressInfo & progress, int iterations, CudaMatrix<cudafloat> & inputData, CudaArray<int> & targets, CudaMatrix3D<cudafloat> & weights, CudaMatrix<int> & mapView, CudaArray<int> & winNode, cudafloat mapRadius, cudafloat timeConstant, LogHTML & summaryLog, LogHTML & log, int & tools, int & maptype);

		void FindBestMatchingUnit(int vector, CudaMatrix<cudafloat> & inputData, CudaArray<int> & targets, CudaMatrix3D<Features> & weights, CudaMatrix3D<int> & mapView, CudaArray<int> & winNode);

		cudafloat CalculateDistance(int input, int wx, int wy, int wz, CudaMatrix<cudafloat> & inputData, CudaMatrix3D<Features> & weights);

		void InitWeights(CudaMatrix3D<cudafloat> & weights, int & tools, int maxScale);

		void NormalizeWeights(CudaMatrix3D<Features> & weights);

		void WriteWeights(CudaMatrix3D<Features> & weights, char * weightsOutput);

		void ShowMapView(LogHTML & log, CudaMatrix3D<int> & mapView, char * mapOutput);

		void WritePLYFile(CudaMatrix3D<cudafloat> & weights, CudaMatrix<int> & mapView, int mapz, char * plyOutput);

		int getMapLocation(int row, int col, int mapx);
	};

} // namespace GPUMLib

#endif // GPUMLIB_SOM_WIDGET_H
