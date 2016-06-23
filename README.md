# Commons Weka (BETA) #

[![Build Status](https://img.shields.io/travis/octavian-h/commons-weka/master.svg)](https://travis-ci.org/octavian-h/commons-weka)
[![Coverage Status](https://img.shields.io/coveralls/octavian-h/commons-weka/master.svg)](https://coveralls.io/github/octavian-h/commons-weka?branch=master)

## Overview ##
The goal of this utility is to provide optimised functions to run validation tasks using Weka library.

## Features ##

* Multi-threaded evaluation comparison of variable train set size
* Hold one out validation

## Usage ##
Add the following dependency to your maven project.
```xml
<dependency>
    <groupId>ro.hasna.commons</groupId>
    <artifactId>commons-weka</artifactId>
    <version>0.3</version>
</dependency>
```

And also add the following custom repository.
```xml
<repository>
    <snapshots>
        <enabled>false</enabled>
    </snapshots>
    <id>central</id>
    <name>bintray</name>
    <url>http://jcenter.bintray.com</url>
</repository>
```

### Code Example ###

```java
class Test{
    public static void main(String[] args){
        Classifier classifier = new IBk(1); //1NN classifier
        Instances train = WekaUtils.readInstances("path/to/train.arff");
        Instances test = WekaUtils.readInstances("path/to/test.arff");

        MultipleTrainTestValidation task = new MultipleTrainTestValidation.Builder(classifier, train, test)
                                          .trainSizePercentages(Arrays.asList(0.6, 0.7, 0.8))
                                          .iterations(10)
                                          .build();
        List<EvaluationResult> results = task.call();

        ValidationResultWriter writer = new CsvValidationResultWriter("path/to/result.csv")
                                          .resultMetadataKeys(MultipleTrainTestValidation.RESULT_METADATA_KEYS)
                                          .writeConfusionMatrix(true)
                                          .numClasses(train.numClasses())
                                          .writeHeader(true)
                                          .build();
        writer.write(results);
        writer.close();
    }
}
```
## Planned features ##

* Multi-threaded cross-domain evaluation comparison ...

## Contributing ##

* Create your own fork of [octavian-h/commons-weka](https://github.com/octavian-h/commons-weka)
* Make the changes
* Submit a pull request

## Licensing ##
_Commons Weka_ utility is licensed under the Apache License, Version 2.0.
See [LICENSE](LICENSE.txt) for the full license text. 
