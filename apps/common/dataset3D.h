#ifndef GPUMLIB_DATASET3D_H
#define GPUMLIB_DATASET3D_H

#include "dataset.h"

#include "data3DInputFile.h"

namespace GPUMLib {
  class Dataset3D : public Dataset {

    private:
      void Dataset3D::Init(const QString & filename, bool hasHeader, int nInputs, int nOutputs, int maximumSamples, LogHTML & log, const Properties * trainProperties, bool rescale, cudafloat rescaleMin, cudafloat rescaleMax) {
        using namespace std;

        Data3DInputFile f(filename);

        int nVars = nInputs + nOutputs;

        if (hasHeader) {
          properties.headers = f.ReadLine();

          if (properties.headers.length() != nVars) {
            throw QString(QObject::tr("The header contains %1 variables. However %2 variables were expected (%3 inputs + %4 outputs).")).arg(properties.headers.length()).arg(nVars).arg(nInputs).arg(nOutputs);
          }
        } else if (trainProperties == nullptr) {
          for (int i = 1; i <= nInputs; i++) properties.headers.append(QString("input %1").arg(i));
          for (int o = 1; o <= nOutputs; o++) properties.headers.append(QString("output %1").arg(o));
        }

        if (trainProperties == nullptr) {
          properties.min.ResizeWithoutPreservingData(nVars);
          properties.max.ResizeWithoutPreservingData(nVars);
        } else {
          properties.headers = trainProperties->headers;
          properties.min = trainProperties->min;
          properties.max = trainProperties->max;
        }

        properties.varContainsMissingValues.ResizeWithoutPreservingData(nInputs);
        for (int i = 0; i < nInputs; i++) properties.varContainsMissingValues[i] = false;

        int samples = 0;
        std::vector<cudafloat> data;

        while (!f.AtEnd()) {
          QStringList vars = f.ReadLine();

          if (vars.length() == 0) continue;

          if (vars.length() != nVars) {
            throw QString(QObject::tr("Sample %1 contains %2 variables. However %3 variables were expected (%4 inputs + %5 outputs).")).arg(samples + 1).arg(vars.length()).arg(nVars).arg(nInputs).arg(nOutputs);
          }

          for (int i = 0; i < nVars; ++i) {
            QString v = vars[i].trimmed();

            bool conversionOk;
            cudafloat value = (cudafloat)v.toDouble(&conversionOk);
            if (!conversionOk) {
              if (v.isEmpty() || v == "?") {
                value = numeric_limits<cudafloat>::quiet_NaN();
              } else {
                QString error = QString(QObject::tr("Invalid value (<i>%1</i>) detected in sample %2, %3.")).arg(v).arg(samples + 1).arg(properties.headers[i]);
                if (!hasHeader && samples == 0) error += QObject::tr(" Check if your dataset contains a header line. If so, please check the <i>header line</i> option in the <i>Datasets</i> properties.");
                throw error;
              }
            }

            if (GPUMLib::IsInfOrNaN(value)) {
              if (i < nInputs) {
                properties.varContainsMissingValues[i] = true;

                if (trainProperties != nullptr && !trainProperties->varContainsMissingValues[i]) {
                  throw QString(QObject::tr("Missing values were detected in a variable which contains no missing values in the training dataset (in sample %1, %2).")).arg(samples + 1).arg(properties.headers[i]);
                }
              } else {
                throw QString(QObject::tr("Missing values not allowed for output variables (sample %1, %2).")).arg(samples + 1).arg(properties.headers[i]);
              }
            } else if (trainProperties == nullptr) {
              if (samples == 0) {
                properties.max[i] = properties.min[i] = value;
              } else if (value < properties.min[i]) {
                properties.min[i] = value;
              } else if (value > properties.max[i]) {
                properties.max[i] = value;
              }
            }

            data.push_back(value);
          }

          if (++samples == maximumSamples) break;
        }

        if (samples == 0) {
          throw QString(QObject::tr("The dataset does not contain any data."));
        } else if (samples < maximumSamples) {
          throw QString(QObject::tr("Could not read all samples. Only %1 out of %2 samples were read. To read all samples set <i>Samples</i> to zero.")).arg(samples).arg(maximumSamples);
        }

        if (trainProperties == nullptr) {
          bool warning = false;

          for (int i = 0; i < nVars; i++) {
            if (GPUMLib::AbsDiff(properties.max[i], properties.min[i]) < cudafloat(0.0000001)) {
              //todo: use FLT_DIG in cfloat
              //properties.max[i] = properties.min[i];

              if (!warning) {
                log.Append("<div style=\"margin-top:1ex;\">");
                log.AppendTag("span", "<b>WARNING:</b>", "color:red");
                log.Append(" The following variables are always constant. Using them may result in inadequate models:</div>");
                log.BeginList();

                warning = true;
              }

              log.AddListItem(properties.headers[i]);
            }
          }

          if (warning) log.EndList();
        }

        inputs.ResizeWithoutPreservingData(samples, nInputs);
        targets.ResizeWithoutPreservingData(samples, nOutputs);

        auto iter = data.cbegin();
        for (int s = 0; s < samples; s++) {
          for (int i = 0; i < nInputs; i++) {
            cudafloat v = *iter++;

            if (rescale && !GPUMLib::IsInfOrNaN(v)) {
              if (properties.min[i] == properties.max[i]) {
                if (v < rescaleMin) {
                  v = rescaleMin;
                } else if (v > rescaleMax) {
                  v = rescaleMax;
                }
              } else {
                v = rescaleMin + (rescaleMax - rescaleMin) * (v - properties.min[i]) / (properties.max[i] - properties.min[i]);
              }
            }

            inputs(s, i) = v;
          }

          for (int o = 0; o < nOutputs; o++) {
            int var = nInputs + o;

            cudafloat v = *iter++;

            if (rescale) {
              if (properties.min[var] == properties.max[var]) {
                if (v < 0) {
                  v = 0;
                } else if (v > 1) {
                  v = 1;
                }
              } else {
                v = (v - properties.min[var]) / (properties.max[var] - properties.min[var]);
              }
            }

            targets(s, o) = v;
          }
        }
      }

    public:
      Dataset3D(const QString & filename, bool hasHeader, bool rescale, int nInputs, int nOutputs, int maximumSamples, LogHTML & log, const Properties * trainProperties = nullptr) {
  		Init(filename, hasHeader, nInputs, nOutputs, maximumSamples, log, trainProperties, rescale, -1, 1);
  	  }

  	  Dataset3D(const QString & filename, bool hasHeader, cudafloat rescaleMin, cudafloat rescaleMax, int nInputs, int nOutputs, int maximumSamples, LogHTML & log, const Properties * trainProperties = nullptr) {
  		Init(filename, hasHeader, nInputs, nOutputs, maximumSamples, log, trainProperties, true, rescaleMin, rescaleMax);
  	  }
  };
}

#endif // GPUMLIB_DATASET3D_H
