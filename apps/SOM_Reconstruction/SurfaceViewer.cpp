#include "SurfaceViewer.h"

void SurfaceViewer::initViewer(const int RESCALE_MAX) {
	viewer->setBackgroundColor(0.8, 0.89, 1);
	viewer->initCameraParameters();
	viewer->setCameraPosition(RESCALE_MAX / 2, RESCALE_MAX / 2, RESCALE_MAX * -2, RESCALE_MAX / 2, RESCALE_MAX / 2, RESCALE_MAX / 2, 0.00586493, 0.998639, -0.0518208);
}

void SurfaceViewer::loadMesh(std::string meshPath) {
	pcl::io::load(meshPath, mesh);
	viewer->addPolygonMesh(mesh, "meshes", 0);
}

void SurfaceViewer::run() {
	while (!viewer->wasStopped()){
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}
