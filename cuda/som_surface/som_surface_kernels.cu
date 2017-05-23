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

#include "som_surface_kernels.h"

__global__ void ComputeDistancesSOMkernel(cudafloat * inputData, cudafloat * weights, int vector, int features, cudafloat * distances) {
	extern __shared__ cudafloat sdist [];

	int w = (blockIdx.y * gridDim.x + blockIdx.x);

	cudafloat distance = 0.0;

	for (int f = threadIdx.x; f < features; f += blockDim.x) {
		cudafloat fdist = inputData[vector * features + f] - weights[w * features + f];
		distance += fdist * fdist;
	}

	sdist[threadIdx.x] = distance;

	__syncthreads();

	// reduction
	for (int dist = blockDim.x; dist >= 2;) {
		dist /= 2;
		if (threadIdx.x < dist) {
			sdist[threadIdx.x] += sdist[threadIdx.x + dist];
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		distances[w] = sqrt(sdist[0]);
	}
}

cudaError_t ComputeDistancesSOM(dim3 gridSize, int blockSize, cudafloat * inputData, cudafloat * weights, int vector, int numberFeatures, cudafloat * distances) {
	ComputeDistancesSOMkernel<<<gridSize, blockSize, blockSize * sizeof(cudafloat)>>>(inputData, weights, vector, numberFeatures, distances);

	return cudaGetLastError();
}

__global__ void UpdateWeightsSOMkernel(int * bmu, int * mapView, int mapx, int mapy, cudafloat * inputData, int vector, int features, int target, cudafloat neighbourhoodRadiusSquare, cudafloat * weights, cudafloat learningRate) {
	__shared__ int winx;
	__shared__ int winy;

	if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
		winx = *bmu % mapx;
		winy = *bmu / mapx;
		mapView[*bmu] = target;
	}
	__syncthreads();

	for (int y = threadIdx.z; y < mapy; y += blockDim.z) {
		cudafloat dy = winy - y;

		for (int x = threadIdx.y; x < mapx; x += blockDim.y) {
			cudafloat dx = winx - x;

			cudafloat distance = dx * dx + dy * dy;

			cudafloat influence = exp(-distance / (2 * neighbourhoodRadiusSquare));

			if (distance < neighbourhoodRadiusSquare) {
				for (int f = threadIdx.x; f < features; f += blockDim.x) {
					int idx = (y * mapx + x) * features + f;

					weights[idx] += learningRate * influence * (inputData[vector * features + f] - weights[idx]);
				}
			}
		}
	}
}

cudaError_t UpdateWeightsSOM(dim3 blockSize, int * bmu, int * mapView, int mapx, int mapy, cudafloat * inputData, int vector, int features, int target, cudafloat neighbourhoodRadiusSquared, cudafloat * weights, cudafloat learningRate) {
	UpdateWeightsSOMkernel<<<1, blockSize>>>(bmu, mapView, mapx, mapy, inputData, vector, features, target, neighbourhoodRadiusSquared, weights, learningRate);

	return cudaGetLastError();
}

__global__ void UpdateWeightsSOMDualkernel(int * bmu, int * mapView, int mapx, int mapy, cudafloat * inputData, int vector, int features, int target, cudafloat neighbourhoodRadiusSquare, cudafloat * weights, cudafloat learningRate) {
	__shared__ int winx;
	__shared__ int winy;

	if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
		winx = *bmu % mapx;
		winy = *bmu / mapx;
		mapView[*bmu] = target;
	}
	__syncthreads();

	for (int y = threadIdx.z; y < mapy; y += blockDim.z) {
		cudafloat distance = 0;
		cudafloat dz = abs((int)(winy / (mapy / 2)) - (int)(y / (mapy / 2)));
		cudafloat dy = 0;
		cudafloat leftDist = 0;
		cudafloat rightDist = 0;
		cudafloat topDist = 0;
		cudafloat botDist = 0;
		int tempY = y >= (mapy / 2) ? y - (mapy / 2) : y;
		int tempwinY = winy >= (mapy / 2) ? winy - (mapy / 2) : winy;

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


		for (int x = threadIdx.y; x < mapx; x += blockDim.y) {
			cudafloat dx = 0;

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
				distance = leftDist < rightDist ? leftDist : rightDist;
				distance = distance < topDist ? distance : topDist;
				distance = distance < botDist ? distance : botDist;
			}
			else {
				distance = dx * dx + dy * dy;
			}

			cudafloat influence = exp(-distance / (2 * neighbourhoodRadiusSquare));

			if (distance < neighbourhoodRadiusSquare) {
				for (int f = threadIdx.x; f < features; f += blockDim.x) {
					int idx = (y * mapx + x) * features + f;

					weights[idx] += learningRate * influence * (inputData[vector * features + f] - weights[idx]);
				}
			}
		}
	}
}

cudaError_t UpdateWeightsSOMDual(dim3 blockSize, int * bmu, int * mapView, int mapx, int mapy, cudafloat * inputData, int vector, int features, int target, cudafloat neighbourhoodRadiusSquared, cudafloat * weights, cudafloat learningRate) {
	UpdateWeightsSOMDualkernel << <1, blockSize >> >(bmu, mapView, mapx, mapy, inputData, vector, features, target, neighbourhoodRadiusSquared, weights, learningRate);

	return cudaGetLastError();
}
