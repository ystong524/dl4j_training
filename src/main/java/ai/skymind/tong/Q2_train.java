package ai.skymind.tong;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class Q2_train {
    final static int seed = 1234;
    final static int batchSize = 5000;
    final static int labelIndex = 784;
    final static int numClasses = 10;
    final static int epoch = 5;

    public static void main(String[] args) throws IOException, InterruptedException {
        File f = new File("C:/Users/user/Desktop/Penjana_DLPC/Day11/Q2/mnist_784_csv.csv");

        RecordReader rr = new CSVRecordReader(1, ',');
        rr.initialize(new FileSplit(f));

        //loop data in
        DataSetIterator iterator = new RecordReaderDataSetIterator(rr, batchSize, labelIndex, numClasses);
        List<List<Writable>> to_train = new ArrayList<>();
        List<List<Writable>> to_test = new ArrayList<>();
        //split test-train here because large dataset
        int j = 0;
        while (rr.hasNext())
        {
            if (j%10 <= 8) to_train.add(rr.next());  //90% train
            else to_test.add(rr.next());
            j++;
        }
        System.out.println("training numExamples: " + to_train.size());
        System.out.println("testing numExamples: " + to_test.size());
        System.out.println("total numExamples: " + (to_train.size() + to_test.size()));

        //turn list to collectionRR
        RecordReader crr1 = new CollectionRecordReader(to_train);
        RecordReader crr2 = new CollectionRecordReader(to_test);

        //turn it into Iterator, now with datasize known, can load altogether or by batchsize
        DataSetIterator train_iter =  new RecordReaderDataSetIterator(crr1, batchSize, labelIndex, numClasses);
        DataSetIterator test_iter =  new RecordReaderDataSetIterator(crr2, batchSize, labelIndex, numClasses);
        /*//now we can split
        DataSet allData =all_iter.next();
        System.out.println("allData.numExamples() = " + allData.numExamples());
        System.out.println("allData.numInputs() = " + allData.numInputs());
        System.out.println("allData.numOutcomes() = " + allData.numOutcomes());

        //manipulate data
        allData.shuffle();
        //split for test-train
        SplitTestAndTrain test_train = allData.splitTestAndTrain(0.8);
        DataSet train = test_train.getTrain();
        DataSet test = test_train.getTest();

        System.out.println("train.numExamples() = " + train.numExamples());
        System.out.println("test.numExamples() = " + test.numExamples());*/


        //preprocessing
        NormalizerMinMaxScaler scaler = new NormalizerMinMaxScaler(0, 1);
        scaler.fit(train_iter);

        //apply for both datasets
        train_iter.setPreProcessor(scaler);
        test_iter.setPreProcessor(scaler);
        System.out.println("input column = " + train_iter.inputColumns());

        HashMap<Integer, Double> schedule = new HashMap<>();
        schedule.put(0, 1e-3);
        schedule.put(8, 1e-4);
        //schedule.put(, 1e-5);

        //setting model config
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(new MapSchedule(ScheduleType.EPOCH, schedule)))  //optimizer
                .weightInit(WeightInit.XAVIER)  //weight initialization
                .activation(Activation.RELU)  //default activation function
                .l2(0.001)
                .list()
                .layer(new DenseLayer.Builder()  //input layer
                        .nIn(train_iter.inputColumns())  //specify the no. of examples in training data
                        .nOut(224)
                        .build())
                .layer(new BatchNormalization())
                .layer(new DenseLayer.Builder()  //hidden layer 1
                        .nOut(282)
                        .build())
                .layer(new BatchNormalization())
                .layer(new DenseLayer.Builder()  //hidden layer 1
                        .nOut(424)
                        .build())
                .layer(new DenseLayer.Builder()  //hidden layer 1
                        .nOut(512)
                        .build())
                .layer(new OutputLayer.Builder()  //output layer
                        .lossFunction(LossFunctions.LossFunction.MCXENT)  //multiclass cross-entropy loss
                        .activation(Activation.SOFTMAX)
                        .nOut(train_iter.totalOutcomes())  //specifiy the class nunmber for datasets
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config); //construct model
        model.init();  //initialize model

        //set up server to record training statistics
        InMemoryStatsStorage storage = new InMemoryStatsStorage(); //allocate some memory
        UIServer server = UIServer.getInstance();
        server.attach(storage);  //attach memory to use for server
        model.setListeners(new StatsListener(storage), new ScoreIterationListener(1000));  //print per 1000 iteration

        //training model
        int epoch = 3;
        for (int i = 0; i < epoch; ++i)
        {
            model.fit(train_iter);
            Evaluation evalTrain = model.evaluate(train_iter);  //validation with training data
            System.out.println("\nepoch = " + i);
            System.out.println("Training accuracy: " + evalTrain.accuracy());
            Evaluation evalTest = model.evaluate(test_iter);  //test with testing data

            System.out.println("Test  accuracy: " + evalTest.accuracy());

        }

        Evaluation evalTrain = model.evaluate(train_iter);  //validation with training data
        Evaluation evalTest = model.evaluate(test_iter);  //test with testing data
        System.out.println("Training: " + evalTrain.stats());
        System.out.println("Testing: " + evalTest.stats());


        System.out.println(evalTrain.confusionMatrix());
    }
}
