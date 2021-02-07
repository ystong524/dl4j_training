package ai.skymind.DataVec;

import org.apache.commons.io.FileUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.writer.RecordWriter;
import org.datavec.api.records.writer.impl.csv.CSVRecordWriter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.partition.NumberOfRecordsPartitioner;
import org.datavec.api.split.partition.Partitioner;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.CategoricalColumnCondition;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.integer.ReplaceInvalidWithIntegerTransform;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.datavec.local.transforms.LocalTransformExecutor;

import java.io.File;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class load_csv_demo {
    private static File inputFile;
    public static void main(String[] args) throws Exception {
        inputFile = new ClassPathResource("datavec/titanic/train.csv").getFile();

        /*
        Exercise 1: Prepare titanic dataset
        - read csv file
        - define schema
        - define transform process
        - apply transform process
        - split into training and test dataset
        - normalize data
         */

        // read csv file
        FileSplit fs = new FileSplit(inputFile);
        RecordReader rr = new CSVRecordReader(1, ',');
        rr.initialize(fs);

        //PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
        // define schema
        Schema sch = new Schema.Builder()
                .addColumnInteger("PassengerId")
                .addColumnCategorical("Survived", Arrays.asList("0", "1"))
                .addColumnCategorical("PcClass", Arrays.asList("1", "2", "3"))
                .addColumnString("Name")
                .addColumnCategorical("Sex", Arrays.asList("male", "female"))
                .addColumnInteger("Age")
                .addColumnInteger("SibSp")
                .addColumnInteger("Parch")
                .addColumnString("Ticket")
                .addColumnDouble("Fare")
                .addColumnString("Cabin")
                .addColumnCategorical("Embarked", Arrays.asList("S", "C", "Q"))
                .build();

        System.out.println(sch);

        System.out.println("no. of cols: " + sch.numColumns());
        System.out.println("col names: " + sch.getColumnNames());
        System.out.println("col types: " + sch.getColumnTypes());


        // cabin has 687 missing values
        // embarked has 2 missing values
        // age has 177 missing values
        // define transform process
        /*
        - remove unused column (PassengerId, Name, Tiket, Cabin)
        - replace Age missing value with 0. hint: use .transform(new ReplaceInvalidWithIntegerTransform(...))
        - replace Embarked missing value with "S". hint: use .conditionalReplaceValueTransform(...))
        - convert Sex(category) to integer
        - convert Pclass, Embarked to one hot encoding
        */

        TransformProcess tp = new TransformProcess.Builder(sch)
                .removeColumns("PassengerId", "Name", "Ticket", "Cabin")
                .transform(new ReplaceInvalidWithIntegerTransform("Age", 0))
                .conditionalReplaceValueTransform("Embarked", new Text("S"),
                        new CategoricalColumnCondition("Embarked", ConditionOp.Equal, ""))
                //.transform(new ReplaceEmptyStringTransform("Embarked", new Text("S")))
                .categoricalToInteger("Sex")
                .categoricalToOneHot("PcClass", "Embarked")
                .build();

        Schema finalsch = tp.getFinalSchema();
        System.out.println("After transformantion:\n" + finalsch);


        //before is illustration by schema and transforamtion process to schema
        //now is to loop out the data to vectors

        //Process the data:  **loop data into collection
        List<List<Writable>> originalData = new ArrayList<>();
        while(rr.hasNext()){
            originalData.add(rr.next());
        }

        //Apply transform process
        List<List<Writable>> transData = LocalTransformExecutor.execute(originalData, tp);

        //Create iterator from processedData
        RecordReader collectionRecordReader = new CollectionRecordReader(transData);  //it's a collection
        DataSetIterator iterator = new RecordReaderDataSetIterator(collectionRecordReader, transData.size(),0,2);
        DataSet allData = iterator.next();

        //Shuffle and split data into training and test dataset
        allData.shuffle();
        SplitTestAndTrain test_train = allData.splitTestAndTrain(0.8);
        DataSet trainingData = test_train.getTrain();
        DataSet testData = test_train.getTest();

        //Create iterator for splitted training and test dataset
        DataSetIterator trainIterator = new ViewIterator(trainingData, 4);
        DataSetIterator testIterator = new ViewIterator(testData, 2);

        //Normalize data to 0 - 1
        DataNormalization scaler = new NormalizerMinMaxScaler(0, 1);
        scaler.fit(trainIterator);

        trainIterator.setPreProcessor(scaler);
        testIterator.setPreProcessor(scaler);

        System.out.println("Sample of training vector: \n"+ trainIterator.next());

        writeAndPrint(transData);
    }

    private static void writeAndPrint(List<List<Writable>> processedData) throws Exception {
        // write processed data into file
        RecordWriter rw = new CSVRecordWriter();
        File outputFile = new File("titanicTransform.csv");
        if(outputFile.exists()) outputFile.delete();
        outputFile.createNewFile();

        Partitioner p = new NumberOfRecordsPartitioner();
        rw.initialize(new FileSplit(outputFile), p);
        rw.writeBatch(processedData);
        rw.close();

        //Print before + after:
        System.out.println("\n\n---- Original Data File ----");
        String originalFileContents = FileUtils.readFileToString(inputFile, Charset.defaultCharset());
        System.out.println(originalFileContents);

        System.out.println("\n\n---- Processed Data File ----");
        String fileContents = FileUtils.readFileToString(outputFile, Charset.defaultCharset());
        System.out.println(fileContents);
    }
}
