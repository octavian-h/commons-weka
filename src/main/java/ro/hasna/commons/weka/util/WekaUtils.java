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
package ro.hasna.commons.weka.util;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

/**
 * @since 0.1
 */
public class WekaUtils {
    /**
     * Private constructor.
     */
    private WekaUtils() {
    }

    /**
     * Read the instances from an ARFF file.
     *
     * @param inputPath the path to the ARFF file
     * @return the instances
     * @throws IOException if the file couldn't be read
     */
    public static Instances readInstances(Path inputPath) throws IOException {
        BufferedReader reader = Files.newBufferedReader(inputPath, StandardCharsets.UTF_8);
        ArffLoader.ArffReader arff = new ArffLoader.ArffReader(reader);
        Instances data = arff.getData();
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    /**
     * Write the instances to an ARFF file.
     *
     * @param instances  the data set
     * @param outputPath the path to the ARFF file
     */
    public static void writeInstances(Instances instances, Path outputPath) throws IOException {
        ArffSaver saver = new ArffSaver();
        saver.setInstances(instances);
        saver.setFile(new File(outputPath.toUri()));
        saver.writeBatch();
    }


    /**
     * Divide the data in two sets: train and test.
     *
     * @param instances                the data set
     * @param trainInstancesPercentage the percentage instances included in the train set
     * @return an array with two elements: train and test instances
     */
    public static Instances[] getTrainAndTestInstancesStratified(Instances instances, double trainInstancesPercentage) {
        Instances[] result = new Instances[2];
        int initialSize = instances.size();
        int trainSize = (int) (initialSize * trainInstancesPercentage);
        if (trainSize == 0) {
            trainSize = 1;
        }
        result[0] = new Instances(instances, trainSize);
        result[1] = new Instances(instances, initialSize - trainSize);

        Map<Double, Integer> f = getClassesDistribution(instances);
        for (Map.Entry<Double, Integer> entry : f.entrySet()) {
            Integer value = entry.getValue();
            int newValue = (int) (value * trainInstancesPercentage);
            if (newValue == 0) {
                newValue = 1;
            }
            entry.setValue(newValue);
        }
        for (Instance instance : instances) {
            double key = instance.classValue();
            Integer nr = f.get(key);
            if (nr > 0) {
                //add to train set
                result[0].add(instance);
                //decrease frequency
                f.put(key, nr - 1);
            } else {
                //add to test set
                result[1].add(instance);
            }
        }
        return result;
    }

    /**
     * Compute the distribution of the classes from the instances.
     *
     * @param instances the data set
     * @return a map with pairs of (class id, apparition count)
     */
    public static Map<Double, Integer> getClassesDistribution(Instances instances) {
        Map<Double, Integer> f = new HashMap<>();
        for (Instance instance : instances) {
            double key = instance.classValue();
            Integer nr = f.get(key);
            if (nr == null) {
                nr = 0;
            }
            f.put(key, nr + 1);
        }
        return f;
    }

    /**
     * Resample the instances so as to have the same number of instances per class.
     *
     * @param instances the data set
     * @return a balanced data set
     */
    public static Instances getBalancedInstances(Instances instances) {
        Map<Double, Integer> f = getClassesDistribution(instances);
        int instancesPerClass = instances.size();
        for (Integer count : f.values()) {
            if (instancesPerClass > count) {
                instancesPerClass = count;
            }
        }

        Instances result = new Instances(instances, instances.size());
        f = new HashMap<>();
        for (Instance instance : instances) {
            double key = instance.classValue();
            Integer nr = f.get(key);
            if (nr == null) {
                nr = 0;
            }
            if (nr < instancesPerClass) {
                result.add(instance);
                f.put(key, nr + 1);
            }
        }
        return result;
    }
}
