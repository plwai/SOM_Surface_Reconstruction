/*
  SurfaceViewer is a simple mesh viewer by using PCL opensource library.
*/

#ifndef SURFACE_VIEWER
#define SURFACE_VIEWER

#include <pcl/common/common.h>
#include <pcl/io/auto_io.h>
#include <pcl/visualization/pcl_visualizer.h>

class SurfaceViewer {
	private:
		pcl::PolygonMesh mesh;
		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;

	public:
		SurfaceViewer() : viewer(new pcl::visualization::PCLVisualizer("3D Viewer")){}
		void initViewer(const int);
		void loadMesh(std::string);
		void run();
};

#endif
