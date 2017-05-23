#=================================================
# Project GPUMLib (author: Noel Lopes)
#=================================================
# Tested configurations (version 0.3.4):
#
# Windows 7 (x64)
# - Visual Studio 2013 (with Multibyte MFC Library for VS 2013)
# - Qt 5.3
# - CUDA 6.5
# - CULA Dense R17
# - CMake 3.2.2
# - .NET compiler version v4.0.30319
#
# Ubuntu 14.04 LTS
# - gcc version 4.8.2
# - Qt 5
# - CUDA 6.5
# - CULA Dense R18
# - CMake 2.8.12.2
# - Mono 3.2.8
#
# OS X Yosemite version 10.10.3
# - Xcode 6.3.1
# - Qt 5.4
# - CUDA 6.5
# - cmake 3.2.2
# - mono 4.0
#
#=================================================

#-------------------------------------------------
# Project and policies
#-------------------------------------------------
cmake_minimum_required (VERSION 2.8.11 FATAL_ERROR)

if(POLICY CMP0048)
	cmake_policy(SET CMP0048 NEW)
	project(GPUMLib VERSION 0.3.4)
else()
	project(GPUMLib)
	set(PROJECT_VERSION 0.3.4)
endif()

if(POLICY CMP0043)
	cmake_policy(SET CMP0043 NEW)
endif()
#-------------------------------------------------

#-------------------------------------------------
# definitions
#-------------------------------------------------
#add_definitions(-DGPUMLIB_VERSION="\\\"${PROJECT_VERSION} beta\\\"")
#-------------------------------------------------

#-------------------------------------------------
# options
#-------------------------------------------------
option(BUILD_RAN_AND_RBF_NETWORKS "Build RAN and RBF networks (requires CULA)" ON)
option(CULA_R17_OR_ABOVE "CULA version is R17 or above" ON)
option(BUILD_SURFACE_RECONSTRUCTION_SOM "Build surface reconstruction function in SOM (required PCL)" ON)

set(ARCH_OPTIONS default 11 12 13 20 30 35 50 52)
list(GET ARCH_OPTIONS 0 DEFAULT_ARCH)
if (NOT CUDA_ARCH)
	set(CUDA_ARCH ${DEFAULT_ARCH} CACHE STRING "CUDA hardware architecture. The options are: ${ARCH_OPTIONS}" FORCE)
	set_property(CACHE CUDA_ARCH PROPERTY STRINGS ${ARCH_OPTIONS})
endif()
#-------------------------------------------------

#-------------------------------------------------
# Define output directories
#-------------------------------------------------
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
#-------------------------------------------------

#-------------------------------------------------
# Check for 32 or 64 bits system
#-------------------------------------------------
math(EXPR SYSTEM_BITS "${CMAKE_SIZEOF_VOID_P} * 8")
#-------------------------------------------------

#-------------------------------------------------
# C#
#-------------------------------------------------
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR};${CMAKE_MODULE_PATH}")

find_package(CSharp REQUIRED)
include(${CSHARP_USE_FILE})

set(CSHARP_BINARY_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
#-------------------------------------------------

#-------------------------------------------------
# CUDA
#-------------------------------------------------
find_package(CUDA REQUIRED)

if (CUDA_FOUND)
	if((${CUDA_ARCH} MATCHES ${DEFAULT_ARCH}))
		if (CUDA_VERSION_MAJOR LESS 7)
			if (CUDA_VERSION VERSION_LESS 6.5)
				message(FATAL_ERROR "CUDA version ${CUDA_VERSION} found. However, the minimum suported version is 6.5.")
			else()
				set(CUDA_ARCH 11 CACHE STRING "CUDA hardware architecture. The options are: ${ARCH_OPTIONS}" FORCE)
			endif()
		else()
			set(CUDA_ARCH 20 CACHE STRING "CUDA hardware architecture. The options are: ${ARCH_OPTIONS}" FORCE)
		endif()

		message(STATUS "")
		message(STATUS "...:::: WARNING ::::...")
		message(STATUS "The CUDA architecture was not selected.")
		message(STATUS "CUDA code will be optimized for ${CUDA_ARCH} compute capability devices.")
		message(STATUS "...:::: ~~~~~~~ ::::...")
		message(STATUS "")
	endif()

	if (NOT (${CUDA_ARCH} LESS 20))
		add_definitions("-DCUDA_MAX_THREADS_PER_BLOCK=1024")

		if (NOT (${CUDA_ARCH} LESS 30))
			add_definitions("-DCUDA_MAX_GRID_X_DIM=2147483647")
		endif()
	endif()

	list(FIND ARCH_OPTIONS ${CUDA_ARCH} CUDA_ARCH_ID)

	if (${CUDA_ARCH_ID} EQUAL -1)
		message(WARNING "Invalid CUDA architecture. Supported architectures are: ${ARCH_OPTIONS}")
	endif()

	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11;--gpu-architecture=compute_${CUDA_ARCH})
else()
	message(FATAL_ERROR "Could not find CUDA")
endif()

if (APPLE)
	set(CUDA_PROPAGATE_HOST_FLAGS OFF)
endif()
#-------------------------------------------------

#-------------------------------------------------
# Qt
#-------------------------------------------------
if (DEFINED ENV{QTDIR})
	set(CMAKE_PREFIX_PATH $ENV{QTDIR} ${CMAKE_PREFIX_PATH})
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Qt5Core QUIET)
if (NOT Qt5Core_FOUND AND NOT DEFINED ENV{QTDIR})
 	if (WIN32)
 		message(WARNING "The QTDIR (Qt folder) environment variable is not defined. To set this variable, open the control panel, search for environment variables and then set QTDIR to <my-qt-compiler-directory>.")
 	elseif (APPLE)
 		message(WARNING "The QTDIR (Qt folder) environment variable is not defined. To set this variable execute the following command from the terminal: launchctl setenv QTDIR <my-qt-compiler-directory>.")
 	endif()
endif()

find_package(Qt5Widgets REQUIRED)
find_package(Qt5Xml REQUIRED)
#-------------------------------------------------

#-------------------------------------------------
# C++ 11
#-------------------------------------------------
if (NOT MSVC)
	include(CheckCXXCompilerFlag)

	CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_Cx11)

	if (COMPILER_SUPPORTS_Cx11)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
	else()
		message(WARNING "The compiler ${CMAKE_CXX_COMPILER} does not appear to support C++11. Please select a different C++ compiler if that is the case.")
		set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-std=c++11")
	endif()
endif()
#-------------------------------------------------

#-------------------------------------------------
# CULA
#-------------------------------------------------
if (NOT DEFINED ENV{CULA_INC_PATH} AND BUILD_RAN_AND_RBF_NETWORKS)
	message(WARNING "CULA not found. RAN and RBF networks will not be build. If you have CULA installed, setup its environment variables (check the CULA installation guide for more details).")
endif()
#-------------------------------------------------

#-------------------------------------------------
# PCL
#-------------------------------------------------

find_package( PCL 1.8 )

if( PCL_FOUND AND BUILD_SURFACE_RECONSTRUCTION_SOM )
  # [C/C++]>[General]>[Additional Include Directories]
  include_directories( ${PCL_INCLUDE_DIRS} )

  # [C/C++]>[Preprocessor]>[Preprocessor Definitions]
  add_definitions( ${PCL_DEFINITIONS} )

  # For Use Not PreCompiled Features
  #add_definitions( -DPCL_NO_PRECOMPILE )

  # [Linker]>[General]>[Additional Library Directories]
  link_directories( ${PCL_LIBRARY_DIRS} )

  # [Linker]>[Input]>[Additional Dependencies]
  #target_link_libraries( SOM_Reconstruction ${PCL_LIBRARIES} )
else()
  message(WARNING "PCL not found. Surface reconstruction for SOM will not be build")
endif()
#-------------------------------------------------

#-------------------------------------------------
# Auxiliary Functions
#-------------------------------------------------
function(install_library name)
	install(TARGETS ${name} COMPONENT libraries DESTINATION lib)
endfunction()

function(install_example name)
	install(TARGETS ${name} COMPONENT examples DESTINATION examples)
endfunction()

function(install_app name)
	install(TARGETS ${name} COMPONENT apps DESTINATION bin)
endfunction()

function(build_cuda_library name files)
	cuda_add_library(${name} STATIC ${files})
	install_library(${name})

	if (${ARGC} GREATER 2) # include directories
		target_include_directories(${name} PRIVATE ${ARGV2})
	endif()
endfunction()

function(compile_library name files)
	add_library(${name} STATIC ${files})
	install_library(${name})
endfunction()

function(build_library name files)
	compile_library(${name} "${files}")

	if (${ARGC} GREATER 2) # base libraries
		target_link_libraries(${name} ${ARGV2})

		if (${ARGC} GREATER 3) # include directories
			target_include_directories(${name} PRIVATE ${ARGV3})
		endif()
	endif()
endfunction()

function(build_qt_library name files include_dirs)
	set(CMAKE_AUTOMOC ON)

	compile_library(${name} "${files}")
	target_compile_definitions(${name} PRIVATE GPUMLIB_VERSION="${PROJECT_VERSION} beta")
	target_include_directories(${name} PRIVATE ${include_dirs})

	qt5_use_modules(${name} Widgets)
endfunction()

function(build_console_exe name libraries files)
	cuda_add_executable(${name} ${files})
	target_compile_definitions(${name} PRIVATE _CONSOLE)
	target_link_libraries(${name} ${libraries})

	if (${ARGC} GREATER 3) # include directories
		target_include_directories(${name} PRIVATE ${ARGV3})
	endif()
endfunction()

function(build_mfc_app name libraries files)
 	if (MSVC)
 		cuda_add_executable(${name} WIN32 ${files})
		target_compile_definitions(${name} PRIVATE _AFXDLL)
		set(CMAKE_MFC_FLAG 2)
		target_link_libraries(${name} ${libraries})
		install_app(${name})
 	else()
 		message(STATUS "")
 		message(STATUS "...:::: WARNING ::::...")
 		message(STATUS "Currently, the ${name} app is available only for Windows.")
 		message(STATUS "Moreover, it must be compiled with Visual Studio.")
 		message(STATUS "...:::: ~~~~~~~ ::::...")
 		message(STATUS "")
 	endif()
endfunction()

function(build_qt_app name libraries files)
	set(CMAKE_AUTOMOC ON)

	cuda_add_executable(${name} WIN32 MACOSX_BUNDLE ${files})
	target_link_libraries(${name} ${libraries} Qt5::Widgets Qt5::Xml)
	target_include_directories(${name} PRIVATE src/apps/common/qtpropertybrowser/src/)
	install_app(${name})

	qt5_use_modules(${name} Widgets)
endfunction()

#-------------------------------------------------
# Main Functions
#-------------------------------------------------
function(build_cuda_libraries)
	if(BUILD_RAN_AND_RBF_NETWORKS AND DEFINED ENV{CULA_INC_PATH})
		# RBF library
		set (files_rbf
			src/cuda/rbf/rbf_kernels.cu
			src/cuda/kmeans/kmeans_kernels.cu
		)

		build_cuda_library(gpumlib_cuda_rbf "${files_rbf}" $ENV{CULA_INC_PATH})

		# RAN library
		build_cuda_library(gpumlib_cuda_ran src/cuda/ran/ran_kernels.cu $ENV{CULA_INC_PATH})
	endif()

	# BP and MBP library
	set (files_mbp
		src/cuda/mbp/mbp_kernels_rms_error.cu
		src/cuda/mbp/mbp_kernels_correct_weights.cu
		src/cuda/mbp/mbp_kernels_fire_layer.cu
		src/cuda/mbp/mbp_kernels_fire_layer_neurons.cu
		src/cuda/mbp/mbp_kernels_local_gradients.cu
		src/cuda/mbp/mbp_kernels_robust_learning.cu
		src/cuda/mbp/mbp_kernels_fire_selective_inputs.cu
		src/cuda/mbp/mbp_kernels_local_gradients_selective_inputs.cu
		src/cuda/mbp/mbp_kernels_correct_weights_selective_inputs.cu
	)

	build_cuda_library(gpumlib_cuda_mbp "${files_mbp}")

	# NMF library
	set (files_nmf
		src/cuda/nmf/nmf_kernels.cu
		src/cuda/nmf/nmf_kernels_additive_divergence.cu
		src/cuda/nmf/nmf_kernels_quality.cu
	)

	build_cuda_library(gpumlib_cuda_nmf "${files_nmf}")

	# Reduction library
	set (files_reduction
		src/cuda/reduction/reduction_kernels_max.cu
		src/cuda/reduction/reduction_kernels_min.cu
		src/cuda/reduction/reduction_kernels_sum.cu
	)

	build_cuda_library(gpumlib_cuda_reduction "${files_reduction}")

	# RBM and DBN library
	set (files_dbn
		src/cuda/rbm/rbm_kernels_init.cu
		src/cuda/rbm/rbm_kernels_status.cu
		src/cuda/rbm/rbm_kernels_status_small.cu
		src/cuda/rbm/rbm_kernels_correct_weights.cu
	)

	build_cuda_library(gpumlib_cuda_dbn "${files_dbn}")

	# SOM library
	build_cuda_library(gpumlib_cuda_som src/cuda/som/som_kernels.cu)

	# SOM_Reconstruction Specific library
	build_cuda_library(gpumlib_cuda_som_surface src/cuda/som_surface/som_surface_kernels.cu)

	# SVM library
	build_cuda_library(gpumlib_cuda_svm src/cuda/svm/svm_kernels.cu)
endfunction()

function(build_main_libraries)
	if(BUILD_RAN_AND_RBF_NETWORKS AND DEFINED ENV{CULA_INC_PATH})
		# RBF library
		set (files_rbf
			src/RBF/RadialBasisFunction.cpp
			src/RBF/utils.cpp
			src/KMeans/KMeans.cpp
		)

		build_library(gpumlib_rbf "${files_rbf}" gpumlib_cuda_rbf $ENV{CULA_INC_PATH})

		# RAN library
		set (files_ran
			src/RAN/ResourceAllocatingNetwork.cpp
			src/RBF/utils.cpp
		)

		build_library(gpumlib_ran "${files_ran}" gpumlib_cuda_ran $ENV{CULA_INC_PATH})
	endif()

	# BP and MBP library
	set (files_mbp
		src/algorithms/mbp/back_propagation.cpp
		src/algorithms/mbp/multiple_back_propagation.cpp
	)

	build_library(gpumlib_mbp "${files_mbp}" gpumlib_cuda_mbp)

	# NMF library
	set (files_nmf
		src/algorithms/nmf/nmf_multiplicative_divergence.cpp
		src/algorithms/nmf/nmf_multiplicative_euclidian.cpp
	 	src/algorithms/nmf/nmf_additive_divergence.cpp
		src/algorithms/nmf/nmf_additive_euclidian.cpp
	)

	build_library(gpumlib_nmf "${files_nmf}" gpumlib_cuda_nmf)

	# Utils library (Reduction, Random)
	set (files_utils
		src/cuda/reduction/reduction.cpp
		src/cuda/random/random.cpp
	)

	build_library(gpumlib_utils "${files_utils}" gpumlib_cuda_reduction)

	# RBM and DBN library
	build_library(gpumlib_dbn src/RBM/rbm.cpp gpumlib_cuda_dbn)

	# Qt GUI library
	qt5_add_resources(resources_qt_prop_editor
		src/apps/common/qtpropertybrowser/src/qtpropertybrowser.qrc
	)

	set (files_qt_gui
		src/apps/common/qtpropertybrowser/src/qtpropertybrowser.cpp
		src/apps/common/qtpropertybrowser/src/qtpropertymanager.cpp
		src/apps/common/qtpropertybrowser/src/qteditorfactory.cpp
		src/apps/common/qtpropertybrowser/src/qtvariantproperty.cpp
		src/apps/common/qtpropertybrowser/src/qttreepropertybrowser.cpp
		src/apps/common/qtpropertybrowser/src/qtbuttonpropertybrowser.cpp
		src/apps/common/qtpropertybrowser/src/qtgroupboxpropertybrowser.cpp
		src/apps/common/qtpropertybrowser/src/qtpropertybrowserutils.cpp
		src/apps/common/PropertyBrowserExtension/fileedit.cpp
		src/apps/common/PropertyBrowserExtension/filepathmanager.cpp
		src/apps/common/PropertyBrowserExtension/fileeditfactory.cpp
		src/apps/common/ParameterManager/BaseParameter.cpp
		src/apps/common/ParameterManager/BoolParameter.cpp
		src/apps/common/ParameterManager/FilePathParameter.cpp
	 	src/apps/common/ParameterManager/FloatParameter.cpp
		src/apps/common/ParameterManager/GroupParameter.cpp
		src/apps/common/ParameterManager/IntParameter.cpp
		src/apps/common/ParameterManager/ListParameter.cpp
		src/apps/common/ParameterManager/Parameter.cpp
	 	src/apps/common/ParameterManager/ParameterManager.cpp
		src/apps/common/ParameterManager/StringParameter.cpp
		src/apps/common/progress/ProgressInfo.cpp
		src/apps/common/log/LogHTML.cpp
	 	src/apps/common/application/application.cpp
		src/apps/common/widget/AlgorithmWidget.cpp
		${res_qt_prop_editor}
	)

	build_qt_library(gpumlib_gui "${files_qt_gui}" src/apps/common/qtpropertybrowser/src/)
endfunction()

# WARNING: All examples are deprecated since there are applications for all implemented algorithms.
# Currently the code for the examples is still available but it will be removed in future versions.
#If you want to compile the examples remove the comment from the build_examples function call (at the end of the file).
function(build_examples)
 	# BP example
 	set (files_bp
 		src/examples/BP/BP.cpp
 		src/examples/MBP/MBPutils.cpp
 	)
 	build_console_exe(BP gpumlib_mbp "${files_bp}")
	install_example(BP)

 	# MBP example
 	set (files_mbp
 		src/examples/MBP/MBP.cpp
 		src/examples/MBP/MBPutils.cpp
 	)
 	build_console_exe(MBP gpumlib_mbp "${files_mbp}")
	install_example(MBP)

 	# Reduction example
 	build_console_exe(reductionapp gpumlib_utils src/examples/reduction/reductionapp.cpp)
	install_example(reductionapp)
endfunction()

function(build_apps)
 	if (BUILD_RAN_AND_RBF_NETWORKS AND DEFINED ENV{CULA_INC_PATH})
 		# RAN app
 		set (files_ran
 			src/examples/RAN/RAN.cpp
 			src/examples/Dataset/Dataset.cpp
 		)

 		link_directories($ENV{CULA_LIB_PATH_${SYSTEM_BITS}})

 		if (CULA_R17_OR_ABOVE)
 			set (cula_libs cula_lapack_basic)
 		else()
 			set (cula_libs cula_core cula_lapack)
 		endif()

 		set (libs_ran
 			gpumlib_ran
 			${cula_libs}
 			${CUDA_CUBLAS_LIBRARIES}
 		)

 		build_console_exe(RAN "${libs_ran}" "${files_ran}" $ENV{CULA_INC_PATH})
		install_app(RAN)

 		# RBF app
 		set (files_rbf
 			src/examples/RBF/RBF.cpp
 			src/examples/Dataset/Dataset.cpp
 		)

 		set (libs_rbf
 			gpumlib_rbf
 			${cula_libs}
 			${CUDA_CUBLAS_LIBRARIES}
 		)

 		build_console_exe(RBF "${libs_rbf}" "${files_rbf}" $ENV{CULA_INC_PATH})
		install_app(RBF)
 	endif()

	# ATS app
	build_console_exe(ATS gpumlib_mbp src/examples/ATS/ATS.cpp)
	install_app(ATS)

	# DBN app
	set (libs_dbn
		gpumlib_dbn
		gpumlib_utils
		${CUDA_curand_LIBRARY}
	)

	build_console_exe(DBN "${libs_dbn}" src/examples/DBN/DBNapp.cpp)
	install_app(DBN)

	# SVM app
	build_console_exe(SVM gpumlib_cuda_svm src/examples/svm/svm_example.cpp)
	install_app(SVM)

	# NMF (Faces) app
	set (files_nmf
		./src/examples/FacesNMF/FacesNMF.cpp
		./src/examples/FacesNMF/FacesNMFDlg.cpp
		./src/examples/FacesNMF/stdafx.cpp
		./src/examples/FacesNMF/FacesNMF.rc
	)

	set (libs_nmf
		gpumlib_nmf
		${CUDA_CUBLAS_LIBRARIES}
	)

	build_mfc_app(FacesNMF "${libs_nmf}" "${files_nmf}")

	# ATS GUI app
	qt5_add_resources(res_ats
		src/apps/ATS/ATSresources.qrc
	)

	set (files_ats
		src/apps/ATS/main.cpp
		src/apps/ATS/ATSwidget.cpp
		${res_ats}
	)

	set (libs_ats
		gpumlib_mbp
		gpumlib_gui
	)

	build_qt_app(ATSgui "${libs_ats}" "${files_ats}")

	# DBN GUI app
	qt5_add_resources(res_dbn
		src/apps/dbn/dbn_resources.qrc
	)

	set (files_dbn
	 	src/apps/dbn/main.cpp
		src/apps/dbn/dbn_widget.cpp
		${res_dbn}
	)

	set (libs_dbn
		gpumlib_dbn
		gpumlib_utils
		gpumlib_gui
		${CUDA_curand_LIBRARY}
	)

	build_qt_app(DBNgui "${libs_dbn}" "${files_dbn}")

	# SOM GUI app
	qt5_add_resources(res_som
		src/apps/SOM/SOMresources.qrc
	)

	set (files_som
		src/apps/SOM/main.cpp
		src/apps/SOM/SOMwidget.cpp
		${res_som}
	)

	set (libs_som
		gpumlib_cuda_som
		gpumlib_utils
		gpumlib_gui
	)

	build_qt_app(SOM "${libs_som}" "${files_som}")

	if(PCL_FOUND AND BUILD_SURFACE_RECONSTRUCTION_SOM)
		# SOM_Reconstruction GUI app
		qt5_add_resources(res_som_surface
			src/apps/SOM_Reconstruction/SOMSurfaceresources.qrc
		)

		set (files_som_surface
			src/apps/SOM_Reconstruction/main.cpp
			src/apps/SOM_Reconstruction/SOMSurfacewidget.cpp
			${res_som_surface}
		)

		# Build with PCL
		set (libs_som_surface
			gpumlib_cuda_som_surface
			gpumlib_utils
			gpumlib_gui
			${PCL_LIBRARIES}
		)

		build_qt_app(SOM_Reconstruction "${libs_som_surface}" "${files_som_surface}")
	else()
		message(WARNING "PCL not found. Cannot compile SOM for Surface Reconstruction.")
	endif()

	if (CSHARP_FOUND)
		# DBNanalysis app
		set(CSHARP_SOURCE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/examples/DBNanalysis")

		set (files_dbn_analysis
			FormAnalysisDBN.cs
			FormAnalysisDBN.Designer.cs
			DataSet.cs
			DBNmodel.cs
			ModelFilenameEditor.cs
			Program.cs
			RBMlayer.cs
			Settings.cs
			Settings.Designer.cs
			AssemblyInfo.cs
			Resources.Designer.cs
		)

		set (references_dbn_analysis
			System.Drawing.dll
			System.Windows.Forms.dll
			mscorlib.dll
		)

		csharp_add_winapp(DBNanalysis "${references_dbn_analysis}" "${files_dbn_analysis}")
	else()
		message(WARNING "C# compiler (.NET or MONO) not found. Cannot compile DBN analysis tool.")
	endif()
endfunction()

#-------------------------------------------------
# Build
#-------------------------------------------------
build_cuda_libraries()
build_main_libraries()
build_apps()
#build_examples()
