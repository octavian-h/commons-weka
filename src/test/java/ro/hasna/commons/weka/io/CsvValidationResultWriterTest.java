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

import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import ro.hasna.commons.weka.task.MultipleTrainTestValidation;
import ro.hasna.commons.weka.task.TrainTestValidation;
import ro.hasna.commons.weka.type.ValidationResult;
import ro.hasna.commons.weka.util.WekaUtils;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.NumberFormat;
import java.util.*;
import java.util.stream.Collectors;

/**
 * @since 0.3
 */
public class CsvValidationResultWriterTest {
    private static Path TMP_OUTPUT_PATH = Paths.get("tmp_csv_builder.csv");
    @Rule
    public ExpectedException thrown = ExpectedException.none();

    @AfterClass
    public static void removeTmpFile() throws IOException {
        Files.deleteIfExists(TMP_OUTPUT_PATH);
    }

    @Test
    public void testAppend() throws Exception {
        CsvValidationResultWriter writer = new CsvValidationResultWriter.Builder(TMP_OUTPUT_PATH)
                .writeHeader(true)
                .build();
        writer.close();

        writer = new CsvValidationResultWriter.Builder(TMP_OUTPUT_PATH)
                .appendToFile(true)
                .writeHeader(true)
                .build();
        writer.close();

        List<String> lines = Files.readAllLines(TMP_OUTPUT_PATH);
        Assert.assertEquals(2, lines.size());
    }

    @Test
    public void testWrongNumClasses() throws Exception {
        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("numClasses must be strictly positive");

        new CsvValidationResultWriter.Builder(TMP_OUTPUT_PATH)
                .numClasses(0)
                .build();
    }

    @Test
    public void testMissingNumClasses() throws Exception {
        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("numClasses must be provided if writeHeader and writeConfusionMatrix are true");

        new CsvValidationResultWriter.Builder(TMP_OUTPUT_PATH)
                .writeConfusionMatrix(true)
                .writeHeader(true)
                .build();
    }

    @Test
    public void testWrittenHeader() throws Exception {
        CsvValidationResultWriter writer = new CsvValidationResultWriter.Builder(TMP_OUTPUT_PATH)
                .columnDelimiter(';')
                .rowDelimiter("\n")
                .sharedMetadataColumns(Collections.singletonList("flag"))
                .resultMetadataColumns(MultipleTrainTestValidation.RESULT_METADATA_KEYS)
                .writeConfusionMatrix(true)
                .numClasses(3)
                .writeHeader(true)
                .build();
        writer.close();

        List<String[]> values = readCsvFile(TMP_OUTPUT_PATH);
        Assert.assertEquals(1, values.size()); //header
        Assert.assertEquals(14, values.get(0).length); //shared metadata (1) + result metadata (1) + training time + testing time + error + classes * classes
        String[] header = values.get(0);
        String[] expectedHeader = {"flag", "train_size_percentage", "training_time", "testing_time", "classification_error",
                "1_1", "1_2", "1_3", "2_1", "2_2", "2_3", "3_1", "3_2", "3_3"};
        for (int i = 0; i < header.length; i++) {
            Assert.assertEquals(expectedHeader[i], header[i]);
        }
    }

    @Test
    public void testCompleteRun() throws Exception {
        NumberFormat numberFormat = NumberFormat.getNumberInstance(Locale.ENGLISH);
        Classifier classifier = new IBk(1);

        Instances instances = WekaUtils.readInstances(Paths.get(getClass().getResource("/iris.arff").toURI()));
        Instances[] instancesStratified = WekaUtils.getTrainAndTestInstancesStratified(instances, 0.8);
        Instances train = instancesStratified[0];
        train.setRelationName(instances.relationName() + "_TRAIN");
        Instances test = instancesStratified[1];
        test.setRelationName(instances.relationName() + "_TEST");

        ValidationResult validationResult = new TrainTestValidation(classifier, train, test).call();

        Map<String, Object> sharedMetadata = new HashMap<>();
        sharedMetadata.put("classifier", "KNN");
        sharedMetadata.put("train", train.relationName());
        sharedMetadata.put("test", test.relationName());

        CsvValidationResultWriter writer = new CsvValidationResultWriter.Builder(TMP_OUTPUT_PATH)
                .columnDelimiter(';')
                .rowDelimiter("\n")
                .numberFormat(numberFormat)
                .sharedMetadataColumns(Arrays.asList("classifier", "train", "test"))
                .writeConfusionMatrix(true)
                .writeHeader(false)
                .build();
        writer.write(validationResult, sharedMetadata);
        writer.close();

        List<String[]> values = readCsvFile(TMP_OUTPUT_PATH);
        Assert.assertEquals(1, values.size());
        String[] line = values.get(0);
        Assert.assertEquals(15, line.length);
        Assert.assertEquals("KNN", line[0]);
        Assert.assertEquals("iris_TRAIN", line[1]);
        Assert.assertEquals("iris_TEST", line[2]);
        Assert.assertTrue(Long.parseLong(line[3]) > 0); //training_time
        Assert.assertTrue(Long.parseLong(line[4]) > 0); //testing_time
        Assert.assertTrue(Double.parseDouble(line[5]) >= 0); //classification_error
    }

    private List<String[]> readCsvFile(Path path) throws IOException {
        BufferedReader reader = Files.newBufferedReader(path);
        return reader.lines().map(s -> s.split(";")).collect(Collectors.toList());
    }
}