# Commons Weka (BETA) #

## Overview ##
The goal of this utility is to provide optimised functions to run validation tasks using Weka library.

## Features ##

* Multi-threaded evaluation comparison of variable train set size

## Usage ##
Add the following dependency to your maven project.
```xml
<dependency>
    <groupId>ro.hasna.commons.weka</groupId>
    <artifactId>commons-weka</artifactId>
    <version>0.1</version>
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
        try(ValidationResultWriter writer = new CsvWriter("path/to/result.csv").build()){
            MultipleTrainTestValidation task = new MultipleTrainTestValidation.Builder(classifier, train, test, writer)
                                                    .trainSizePercentages(Arrays.asList(0.6, 0.7, 0.8))
                                                    .folds(10)
                                                    .iterations(2)
                                                    .build();
            task.call(); //the method block the current thread until it finish
        }
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
