/*
	Noel Lopes is a Professor at the Polytechnic of Guarda, Portugal
	and a researcher at the CISUC - University of Coimbra, Portugal

	Siti Mariyam Shamsuddin is a Professor at the Faculy of Computing
	Universiti Teknologi Malaysia (UTM), Malaysia and researcher at
	UTM Big Data Centre, Malaysia

	Shafaatunnur Hasan ia a full time researcher at UTM Big Data Centre,
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
	along with this program.  If not, see <http://www.gnu.org/licenses/>.
	*/

#ifndef GPUMLIB_SOM_KERNELS
#define GPUMLIB_SOM_KERNELS

#include <cuda_runtime.h>
#include "../definitions.h"

cudaError_t ComputeDistancesSOM(dim3 gridSize, int blockSize, cudafloat * inputData, cudafloat * weights, int vector, int numberFeatures, cudafloat * distances);
cudaError_t UpdateWeightsSOM(dim3 blockSize, int * bmu, int * mapView, int mapx, int mapy, cudafloat * inputData, int vector, int features, int target, cudafloat neighbourhoodRadiusSquared, cudafloat * weights, cudafloat learningRate);
cudaError_t UpdateWeightsSOMDual(dim3 blockSize, int * bmu, int * mapView, int mapx, int mapy, cudafloat * inputData, int vector, int features, int target, cudafloat neighbourhoodRadiusSquared, cudafloat * weights, cudafloat learningRate);
cudaError_t NormalizeWeightsSOM(dim3 gridSize, int blockSize, int mapx, int mapy, int features, cudafloat * weights);

#endif //GPUMLIB_SOM_KERNELS
