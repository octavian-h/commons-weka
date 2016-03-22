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
package ro.hasna.commons.weka.task;

import org.hamcrest.core.StringContains;
import org.junit.*;
import org.junit.rules.ExpectedException;
import ro.hasna.commons.weka.io.CsvWriter;
import ro.hasna.commons.weka.util.WekaUtils;
import weka.classifiers.Classifier;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.NumberFormat;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;

/**
 * @since 0.1
 */
public class MultipleTrainTestValidationTest {
    private static Path tmpOutputPath = Paths.get("tmp_result.csv");
    @Rule
    public ExpectedException thrown = ExpectedException.none();
    private Classifier classifier;
    private Instances train;
    private Instances test;

    private static List<String[]> readCsvFile(Path path) throws IOException {
        BufferedReader reader = Files.newBufferedReader(path);
        return reader.lines().map(s -> s.split(";")).collect(Collectors.toList());
    }

    @AfterClass
    public static void clearOutputs() throws Exception {
//        Files.deleteIfExists(tmpOutputPath);
    }

    @Before
    public void setUp() throws Exception {
        classifier = new IBk(1);

        Instances instances = WekaUtils.readInstances(Paths.get(getClass().getResource("/iris.arff").toURI()));
        Instances[] instancesStratified = WekaUtils.getTrainAndTestInstancesStratified(instances, 0.8);

        train = instancesStratified[0];
        train.setRelationName(instances.relationName() + "_TRAIN");

        test = instancesStratified[1];
        test.setRelationName(instances.relationName() + "_TEST");
    }

    @After
    public void tearDown() throws Exception {
        classifier = null;
        train = null;
        test = null;
    }

    @Test
    public void testCompleteRun() throws Exception {
        NumberFormat numberFormat = NumberFormat.getNumberInstance(Locale.ENGLISH);
        try (CsvWriter writer = new CsvWriter.Builder(tmpOutputPath)
                .columnDelimiter(';')
                .rowDelimiter("\n")
                .numberFormat(numberFormat)
                .numClasses(train.numClasses())
                .extraColumnsNames(Collections.singletonList("flag"))
                .writeHeaders(true)
                .build()) {

            MultipleTrainTestValidation task = new MultipleTrainTestValidation.Builder(classifier, train, test, writer)
                    .extraColumns(Collections.singletonList("flag_value"))
                    .trainSizePercentages(Arrays.asList(0.6, 0.7, 0.8))
                    .folds(3)
                    .iterations(2)
                    .build();
            task.call();
        }

        List<String[]> values = readCsvFile(tmpOutputPath);
        Assert.assertEquals(19, values.size()); //header + iterations * folds * percentages
        Assert.assertEquals(18, values.get(0).length); //3 + flag + percentage + fold + iteration + 2 + classes * classes
        String[] header = values.get(0);
        String[] expectedHeader = {"classifier", "train", "test", "train_size_percentage", "fold", "iteration", "flag",
                "model_building_time", "predicting_time"};
        for (int i = 0; i < 9; i++) {
            Assert.assertEquals(expectedHeader[i], header[i]);
        }
        for (int i = 1; i < 19; i++) {
            String[] line = values.get(i);
            Assert.assertTrue(line[0].contains("IBk"));
            Assert.assertEquals("iris_TRAIN", line[1]);
            Assert.assertEquals("iris_TEST", line[2]);
            int percentage = (int) (numberFormat.parse(line[3]).doubleValue() * 10);
            Assert.assertTrue(6 <= percentage && percentage <= 8); //train_size_percentage
            Assert.assertTrue(line[4].matches("^[1-3]$")); //fold
            Assert.assertTrue(line[5].matches("^[1-2]$")); //iteration
            Assert.assertEquals("flag_value", line[6]);
            Assert.assertTrue(Long.parseLong(line[7]) > 0); //model_building_time
            Assert.assertTrue(Long.parseLong(line[8]) > 0); //predicting_time
        }
    }

    @Test
    public void testWrongNumberOfIterations() throws Exception {
        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("iterations must be positive");

        new MultipleTrainTestValidation.Builder(classifier, train, test, null)
                .iterations(0)
                .build();
    }

    @Test
    public void testWrongNumberOfFolds() throws Exception {
        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("folds must be positive");

        new MultipleTrainTestValidation.Builder(classifier, train, test, null)
                .folds(0)
                .build();
    }

    @Test
    public void testEmptyPercentagesList() throws Exception {
        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("percentages list is empty");

        new MultipleTrainTestValidation.Builder(classifier, train, test, null)
                .trainSizePercentages(Collections.emptyList())
                .build();
    }

    @Test
    public void testWrongPercentagesValues() throws Exception {
        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("percentages must be between 0 (exclusive) and 1 (inclusive)");

        new MultipleTrainTestValidation.Builder(classifier, train, test, null)
                .trainSizePercentages(Collections.singletonList(0.0))
                .build();
    }

    @Test
    public void testWrongClassifier() throws Exception {
        thrown.expect(ExecutionException.class);
        thrown.expectMessage(StringContains.containsString("Cannot handle multi-valued nominal class!"));

        try (CsvWriter writer = new CsvWriter.Builder(tmpOutputPath).build()) {
            Files.deleteIfExists(tmpOutputPath);

            ExecutorService executorService = Executors.newFixedThreadPool(2);
            LinearRegression regression = new LinearRegression();
            MultipleTrainTestValidation task = new MultipleTrainTestValidation.Builder(regression, train, test, writer)
                    .executorService(executorService)
                    .closeExecutorService(false)
                    .iterations(10)
                    .build();

            task.call();
            executorService.shutdown();
        }
    }
}