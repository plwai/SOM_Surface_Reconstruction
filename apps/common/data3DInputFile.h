#ifndef GPUMLIB_DATA3D_INPUT_FILE_H
#define GPUMLIB_DATA3D_INPUT_FILE_H

#include <QFile>
#include <QTextStream>
#include <QRegularExpression>

namespace GPUMLib {

	class Data3DInputFile {
	private:
		QFile file;
		QTextStream fs;
		QRegularExpression splitExpression;
		int vertexNum, currentVertex;

	public:
		Data3DInputFile(const QString & filename) : file(filename), fs(&file), splitExpression(filename.endsWith(".ply", Qt::CaseInsensitive) ? "\\s+" : "\\s+") {
			if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
				throw QString("Could not open file: <i>%1</i>.").arg(filename);
			}
      else {
        currentVertex = 0;

        for(QString inStr = fs.readLine(); inStr != "end_header"; inStr = fs.readLine()) {
          QStringList vars = inStr.split(splitExpression);

          if(vars.length() == 3 && vars[1] == "vertex") {
            vertexNum = vars[2].toInt();;
          }
        }
      }
		}

		bool AtEnd() {
      currentVertex++;
      if (currentVertex <= vertexNum) {
        return false;
      }
      else{
        return true;
      }
		}

		QStringList ReadLine() {
			QStringList line = fs.readLine().split(splitExpression);
      QStringList coordinates = (QStringList() << line[0] << line[1] << line[2]);
			coordinates.append(QString::number(currentVertex));

			return coordinates;
		}
	};

} // namespace GPUMLib

#endif // DATAINPUTFILE_H
