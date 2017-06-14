# SOM Surface Reconstruction
## What is SOM Surface Reconstruction

SOM Surface Reconstruction is the a developing SOM surface reconstruction extension in GPUMLib.

## Installation

### Dependencies

- [QT](https://www.qt.io/download/)
- [CUDA](https://developer.nvidia.com/cuda-downloads)
- [CMake 3.6.0+](https://cmake.org/download/)
- [GPUMLib](http://gpumlib.sourceforge.net/)
- .Net Compiler

### Optional Dependencies (For PCL Viewer)

- [PCL 1.8 for visual studio 2013](http://unanancyowen.com/en/pcl18/)

### Steps

1. Copy SOM Surface Reconstruction source code into GPUMLib src files
2. Replace the CMakeList.txt at the previous folder. (../src)
3. Configure the code with the generator config "Visual Studio 12 2013" by using CMake 3.6.0+
4. (Optional) Tick PCL Viewer for enabling surface viewer compilation
5. Generate the code
6. Compile the code by using x64 architecture
7. Run the program

### Technical Information

Visit [wiki page](https://github.com/plwai/SOM_Surface_Reconstruction/wiki) for more technical information

## Report an issue

Welcome to report any bug or issue [here](https://github.com/plwai/SOM_Surface_Reconstruction/issues)

## License

GPUMLib is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. 

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
