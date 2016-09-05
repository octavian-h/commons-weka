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

import ro.hasna.commons.weka.type.ValidationResult;
import weka.clusterers.Clusterer;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.Callable;

/**
 * Task for running an external evaluation of a clustering result.
 * <p>
 * <pre>{@code
 *      Clusterer clusterer = ...
 *      Instances instances = WekaUtils.readInstances("path/to/data.arff");
 *      ValidationResult result = new ClusterValidation(clusterer, instances).call();
 *
 *      ValidationResultWriter writer = new CsvValidationResultWriter("path/to/result.csv").build();
 *      writer.write(result);
 *      writer.close();
 * }</pre>
 *
 * @since 0.4
 */
public class ClusterValidation implements Callable<ValidationResult> {
    private final Clusterer clusterer;
    private final Instances instances;

    public ClusterValidation(Clusterer clusterer, Instances instances) {
        this.clusterer = clusterer;
        this.instances = instances;
    }

    @Override
    public ValidationResult call() throws Exception {

        long startTime = System.nanoTime();
        Instances unlabelledInstances = removeInstancesClasses();
        clusterer.buildClusterer(unlabelledInstances);
        long modelBuildingTime = System.nanoTime() - startTime;

        startTime = System.nanoTime();
        int numClasses = instances.numClasses();
        int numberOfClusters = clusterer.numberOfClusters();
        double[][] confusionMatrix = generateInitialConfusionMatrix(numClasses, numberOfClusters, unlabelledInstances);
        int[] clusterClass = computeClusterClass(numClasses, numberOfClusters, confusionMatrix);
        reorderClustersInConfusionMatrix(numClasses, numberOfClusters, confusionMatrix, clusterClass);
        long predictingTime = (System.nanoTime() - startTime) / unlabelledInstances.size();

        return new ValidationResult(modelBuildingTime, predictingTime, confusionMatrix);
    }

    protected Instances removeInstancesClasses() throws Exception {
        Remove filter = new Remove();
        filter.setAttributeIndices("" + (instances.classIndex() + 1));
        filter.setInputFormat(instances);
        return Filter.useFilter(instances, filter);
    }

    protected double[][] generateInitialConfusionMatrix(int numClasses, int numberOfClusters, Instances unlabelledInstances) throws Exception {
        int columns = numClasses > numberOfClusters ? numClasses : numberOfClusters;
        double[][] confusionMatrix = new double[numClasses][columns];
        for (int i = 0; i < unlabelledInstances.size(); i++) {
            Instance instance = unlabelledInstances.get(i);
            int clusterId = clusterer.clusterInstance(instance);
            confusionMatrix[(int) instances.get(i).classValue()][clusterId]++;
        }
        return confusionMatrix;
    }

    protected int[] computeClusterClass(int numClasses, int numberOfClusters, double[][] confusionMatrix) {
        Set<Integer> unassignedClasses = new HashSet<>(numClasses);
        for (int i = 0; i < numClasses; i++) {
            unassignedClasses.add(i);
        }

        Set<Integer> unassignedClusters = new HashSet<>(numberOfClusters);
        int[] clusterClass = new int[numberOfClusters];
        for (int i = 0; i < numberOfClusters; i++) {
            unassignedClusters.add(i);
            clusterClass[i] = -1;
        }

        while (!unassignedClasses.isEmpty() && !unassignedClusters.isEmpty()) {
            double max = -1;
            int maxClass = -1;
            int maxCluster = -1;
            for (Integer unassignedClass : unassignedClasses) {
                for (Integer unassignedCluster : unassignedClusters) {
                    double v = confusionMatrix[unassignedClass][unassignedCluster];
                    if (max < v) {
                        max = v;
                        maxClass = unassignedClass;
                        maxCluster = unassignedCluster;
                    }
                }
            }

            unassignedClasses.remove(maxClass);
            unassignedClusters.remove(maxCluster);
            clusterClass[maxCluster] = maxClass;
        }
        return clusterClass;
    }

    protected void reorderClustersInConfusionMatrix(int numClasses, int numberOfClusters, double[][] confusionMatrix, int[] clusterClass) {
        for (int k = 0; k < numberOfClusters; k++) {
            int p = clusterClass[k];
            if (p != k && p != -1) {
                int aux = clusterClass[p];
                clusterClass[p] = p;
                clusterClass[k] = aux;

                for (int i = 0; i < numClasses; i++) {
                    double tmp = confusionMatrix[i][k];
                    confusionMatrix[i][k] = confusionMatrix[i][p];
                    confusionMatrix[i][p] = tmp;
                }
            }
        }
    }
}
