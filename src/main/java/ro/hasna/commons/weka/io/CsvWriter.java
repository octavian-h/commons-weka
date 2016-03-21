/*
 * Copyright (C) 2016 Octavian Hasna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ro.hasna.commons.weka.io;

import ro.hasna.commons.weka.type.ValidationResult;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;

import java.io.IOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.NumberFormat;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

/**
 * A synchronised CSV writer for the confusion matrices that resulted from the validation.
 *
 * @since 0.1
 */
public class CsvWriter implements ValidationResultWriter {
    private final Writer writer;
    private final char columnDelimiter;
    private final String rowDelimiter;
    private final NumberFormat numberFormat;


    private CsvWriter(Builder builder) {
        writer = builder.writer;
        columnDelimiter = builder.columnDelimiter;
        rowDelimiter = builder.rowDelimiter;
        numberFormat = builder.numberFormat;
    }

    @Override
    public void write(Classifier classifier, Instances trainInstances, Instances testInstances,
                      double trainSizePercentage, int foldNumber, int iterationNumber, List<String> extraColumns,
                      ValidationResult result) throws IOException {

        String classifierName = classifier.getClass().getSimpleName();
        if (classifier instanceof OptionHandler) {
            classifierName += " " + Utils.joinOptions(((OptionHandler) classifier).getOptions());
        }

        StringBuilder sb = new StringBuilder();
        sb.append(classifierName);
        sb.append(columnDelimiter);
        sb.append(trainInstances.relationName());
        sb.append(columnDelimiter);
        sb.append(testInstances.relationName());
        sb.append(columnDelimiter);
        sb.append(numberFormat.format(trainSizePercentage));
        sb.append(columnDelimiter);
        sb.append(foldNumber);
        sb.append(columnDelimiter);
        sb.append(iterationNumber);
        sb.append(columnDelimiter);

        for (String extraColumn : extraColumns) {
            sb.append(extraColumn);
            sb.append(columnDelimiter);
        }

        sb.append(result.getModelBuildingTime());
        sb.append(columnDelimiter);
        sb.append(result.getPredictingTime());
        sb.append(columnDelimiter);

        double[][] confusionMatrix = result.getPredictingResult().confusionMatrix();
        for (double[] row : confusionMatrix) {
            for (double value : row) {
                sb.append(numberFormat.format(value));
                sb.append(columnDelimiter);
            }
        }
        sb.append(rowDelimiter);
        String row = sb.toString();

        synchronized (writer) {
            writer.write(row);
        }
    }

    @Override
    public void flush() throws IOException {
        synchronized (writer) {
            writer.flush();
        }
    }

    @Override
    public void close() throws IOException {
        writer.close();
    }

    public static class Builder {
        private final Path outputPath;
        private Writer writer;
        private char columnDelimiter;
        private String rowDelimiter;
        private NumberFormat numberFormat;
        private List<String> extraColumnsNames;
        private int numClasses;
        private boolean writeHeaders;

        public Builder(Path outputPath) {
            this.outputPath = outputPath;

            // default values
            columnDelimiter = ',';
            rowDelimiter = "\n";
            extraColumnsNames = Collections.emptyList();
            writeHeaders = false;
            numberFormat = NumberFormat.getNumberInstance(Locale.ENGLISH);
        }

        public Builder columnDelimiter(char columnDelimiter) {
            this.columnDelimiter = columnDelimiter;
            return this;
        }

        public Builder rowDelimiter(String rowDelimiter) {
            this.rowDelimiter = rowDelimiter;
            return this;
        }

        public Builder numberFormat(NumberFormat numberFormat) {
            this.numberFormat = numberFormat;
            return this;
        }

        public Builder extraColumnsNames(List<String> extraColumnsNames) {
            this.extraColumnsNames = extraColumnsNames;
            return this;
        }

        public Builder numClasses(int numClasses) {
            this.numClasses = numClasses;
            return this;
        }

        public Builder writeHeaders(boolean writeHeaders) {
            this.writeHeaders = writeHeaders;
            return this;
        }

        public CsvWriter build() throws IOException {
            writer = Files.newBufferedWriter(outputPath, StandardCharsets.UTF_8);

            if (writeHeaders) {
                StringBuilder sb = new StringBuilder();
                sb.append("classifier");
                sb.append(columnDelimiter);
                sb.append("train");
                sb.append(columnDelimiter);
                sb.append("test");
                sb.append(columnDelimiter);
                sb.append("train_size_percentage");
                sb.append(columnDelimiter);
                sb.append("fold");
                sb.append(columnDelimiter);
                sb.append("iteration");
                sb.append(columnDelimiter);

                for (String extraColumnsName : extraColumnsNames) {
                    sb.append(extraColumnsName);
                    sb.append(columnDelimiter);
                }

                sb.append("model_building_time");
                sb.append(columnDelimiter);
                sb.append("predicting_time");
                sb.append(columnDelimiter);

                for (int i = 1; i <= numClasses; i++) {
                    for (int j = 1; j <= numClasses; j++) {
                        sb.append(i);
                        sb.append('_');
                        sb.append(j);
                        sb.append(columnDelimiter);
                    }
                }

                sb.append(rowDelimiter);
                writer.write(sb.toString());
            }

            return new CsvWriter(this);
        }
    }
}
