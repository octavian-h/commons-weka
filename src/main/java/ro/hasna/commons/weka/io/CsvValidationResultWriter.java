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
import ro.hasna.commons.weka.util.WekaUtils;

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
            sb.append(numberFormat.format(WekaUtils.getIncorrectPercentage(confusionMatrix)));
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
            writeHeader = true;
            appendToFile = false;
        }

        /**
         * Configure the CSV column delimiter.
         * The default value is ','.
         *
         * @param delimiter the column delimiter
         * @return a reference to this {@code CsvValidationResultWriter.Builder} object to fulfill the "Builder" pattern
         */
        public Builder columnDelimiter(char delimiter) {
            this.columnDelimiter = delimiter;
            return this;
        }

        /**
         * Configure the CSV row delimiter.
         * The default value is "\n".
         *
         * @param delimiter the row delimiter
         * @return a reference to this {@code CsvValidationResultWriter.Builder} object to fulfill the "Builder" pattern
         */
        public Builder rowDelimiter(String delimiter) {
            this.rowDelimiter = delimiter;
            return this;
        }

        /**
         * Configure the number format to use for writing the numbers.
         * The default format is the one for English language.
         *
         * @param numberFormat the number format
         * @return a reference to this {@code CsvValidationResultWriter.Builder} object to fulfill the "Builder" pattern
         */
        public Builder numberFormat(NumberFormat numberFormat) {
            this.numberFormat = numberFormat;
            return this;
        }

        /**
         * Configure the list of columns from the shared metadata that will be written to the CSV file.
         *
         * @param columns the keys from the shared metadata map
         * @return a reference to this {@code CsvValidationResultWriter.Builder} object to fulfill the "Builder" pattern
         */
        public Builder sharedMetadataColumns(List<String> columns) {
            this.sharedMetadataColumns = columns;
            return this;
        }

        /**
         * Configure the list of columns from the result metadata that will be written to the CSV file.
         *
         * @param columns the keys from the result metadata map
         * @return a reference to this {@code CsvValidationResultWriter.Builder} object to fulfill the "Builder" pattern
         */
        public Builder resultMetadataColumns(List<String> columns) {
            this.resultMetadataColumns = columns;
            return this;
        }

        /**
         * Configure the CSV writer to write the confusion matrix.
         * The confusion matrix will be represented as a one dimensional array.
         * The default value is false.
         *
         * @param writeConfusionMatrix a boolean for writing or not the confusion matrix.
         * @return a reference to this {@code CsvValidationResultWriter.Builder} object to fulfill the "Builder" pattern
         */
        public Builder writeConfusionMatrix(boolean writeConfusionMatrix) {
            this.writeConfusionMatrix = writeConfusionMatrix;
            return this;
        }

        /**
         * Configure the number of classes from the validation result(s).
         * This number must be provided if the CSV writer needs to write the header and the confusion matrix.
         *
         * @param numClasses the number of classes
         * @return a reference to this {@code CsvValidationResultWriter.Builder} object to fulfill the "Builder" pattern
         */
        public Builder numClasses(int numClasses) {
            if (numClasses <= 0) {
                throw new IllegalArgumentException("numClasses must be strictly positive");
            }
            this.numClasses = numClasses;
            return this;
        }

        /**
         * Configure the CSV writer to write the CSV header.
         * The default value is true.
         *
         * @param writeHeaders a boolean for writing or not the header
         * @return a reference to this {@code CsvValidationResultWriter.Builder} object to fulfill the "Builder" pattern
         */
        public Builder writeHeader(boolean writeHeaders) {
            this.writeHeader = writeHeaders;
            return this;
        }

        /**
         * Configure the CSv writer to append to the file.
         * The default value is false which means it (re)creates the output file.
         *
         * @param appendToFile a boolean to append or not the output file
         * @return a reference to this {@code CsvValidationResultWriter.Builder} object to fulfill the "Builder" pattern
         */
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
