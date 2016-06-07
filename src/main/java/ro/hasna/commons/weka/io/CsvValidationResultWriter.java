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

import java.io.IOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.text.NumberFormat;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * A CSV writer for the results of the model validation.
 *
 * @since 0.3
 */
public class CsvValidationResultWriter implements ValidationResultWriter {
    private final Writer writer;
    private final char columnDelimiter;
    private final String rowDelimiter;
    private final NumberFormat numberFormat;
    private final List<String> sharedMetadataKeys;
    private final List<String> resultMetadataKeys;
    private final boolean writeConfusionMatrix;


    private CsvValidationResultWriter(Builder builder) {
        writer = builder.writer;
        columnDelimiter = builder.columnDelimiter;
        rowDelimiter = builder.rowDelimiter;
        numberFormat = builder.numberFormat;
        sharedMetadataKeys = builder.sharedMetadataColumns;
        resultMetadataKeys = builder.resultMetadataColumns;
        writeConfusionMatrix = builder.writeConfusionMatrix;
    }

    @Override
    public void write(List<ValidationResult> results, Map<String, Object> sharedMetadata) throws IOException {
        StringBuilder prefix = new StringBuilder();
        for (String key : sharedMetadataKeys) {
            Object obj = sharedMetadata.get(key);
            if (obj instanceof Number) {
                prefix.append(numberFormat.format(obj));
            } else {
                prefix.append(obj);
            }
            prefix.append(columnDelimiter);
        }

        for (ValidationResult item : results) {
            StringBuilder sb = new StringBuilder();
            sb.append(prefix);

            for (String key : resultMetadataKeys) {
                Object obj = item.getMetadataValue(key);
                if (obj instanceof Number) {
                    sb.append(numberFormat.format(obj));
                } else {
                    sb.append(obj);
                }
                sb.append(columnDelimiter);
            }

            sb.append(item.getTrainingTime());
            sb.append(columnDelimiter);

            sb.append(item.getTestingTime());
            sb.append(columnDelimiter);

            double[][] confusionMatrix = item.getConfusionMatrix();
            double globalSum = 0;
            double firstDiagonalSum = 0;
            for (int i = 0; i < confusionMatrix.length; i++) {
                for (int j = 0; j < confusionMatrix[i].length; j++) {
                    globalSum += confusionMatrix[i][j];
                    if (i == j) {
                        firstDiagonalSum += confusionMatrix[i][j];
                    }
                }
            }
            sb.append(numberFormat.format(1 - firstDiagonalSum / globalSum));
            sb.append(columnDelimiter);

            if (writeConfusionMatrix) {
                for (double[] row : confusionMatrix) {
                    for (double value : row) {
                        sb.append(numberFormat.format(value));
                        sb.append(columnDelimiter);
                    }
                }
            }

            sb.deleteCharAt(sb.length() - 1);
            sb.append(rowDelimiter);

            writer.write(sb.toString());
        }
    }

    @Override
    public void flush() throws IOException {
        writer.flush();
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
        private List<String> sharedMetadataColumns;
        private List<String> resultMetadataColumns;
        private boolean writeConfusionMatrix;
        private int numClasses;
        private boolean writeHeader;
        private boolean appendToFile;

        public Builder(Path outputPath) {
            this.outputPath = outputPath;

            // default values
            columnDelimiter = ',';
            rowDelimiter = "\n";
            numberFormat = NumberFormat.getNumberInstance(Locale.ENGLISH);
            sharedMetadataColumns = Collections.emptyList();
            resultMetadataColumns = Collections.emptyList();
            writeConfusionMatrix = false;
            numClasses = 0;
            writeHeader = false;
            appendToFile = false;
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

        public Builder sharedMetadataColumns(List<String> columnNames) {
            this.sharedMetadataColumns = columnNames;
            return this;
        }

        public Builder resultMetadataColumns(List<String> columnNames) {
            this.resultMetadataColumns = columnNames;
            return this;
        }

        public Builder writeConfusionMatrix(boolean writeConfusionMatrix) {
            this.writeConfusionMatrix = writeConfusionMatrix;
            return this;
        }

        public Builder numClasses(int numClasses) {
            if (numClasses <= 0) {
                throw new IllegalArgumentException("numClasses must be strictly positive");
            }
            this.numClasses = numClasses;
            return this;
        }

        public Builder writeHeader(boolean writeHeaders) {
            this.writeHeader = writeHeaders;
            return this;
        }

        public Builder appendToFile(boolean appendToFile) {
            this.appendToFile = appendToFile;
            return this;
        }

        public CsvValidationResultWriter build() throws IOException {
            if (!appendToFile) {
                writer = Files.newBufferedWriter(outputPath, StandardCharsets.UTF_8);
            } else {
                writer = Files.newBufferedWriter(outputPath, StandardCharsets.UTF_8, StandardOpenOption.APPEND);
            }

            if (writeHeader) {
                if (writeConfusionMatrix && numClasses == 0) {
                    throw new IllegalArgumentException("numClasses must be provided if writeHeader and writeConfusionMatrix are true");
                }

                StringBuilder sb = new StringBuilder();
                for (String key : sharedMetadataColumns) {
                    sb.append(key);
                    sb.append(columnDelimiter);
                }
                for (String key : resultMetadataColumns) {
                    sb.append(key);
                    sb.append(columnDelimiter);
                }

                sb.append("training_time");
                sb.append(columnDelimiter);
                sb.append("testing_time");
                sb.append(columnDelimiter);
                sb.append("classification_error");
                sb.append(columnDelimiter);

                if (writeConfusionMatrix) {
                    for (int i = 1; i <= numClasses; i++) {
                        for (int j = 1; j <= numClasses; j++) {
                            sb.append(i);
                            sb.append('_');
                            sb.append(j);
                            sb.append(columnDelimiter);
                        }
                    }
                }

                sb.deleteCharAt(sb.length() - 1);
                sb.append(rowDelimiter);

                writer.write(sb.toString());
            }

            return new CsvValidationResultWriter(this);
        }
    }
}
