package ai.skymind.tong;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class Q2_train {
    final static int seed = 1234;
    final static int batchSize = 100;
    final static int labelIndex = 784;
    final static int numClasses = 10;
    final static int epoch = 5;

    public static void main(String[] args) throws IOException, InterruptedException {
        File f = new File("C:/Users/user/Desktop/Penjana_DLPC/Day11/Q2/mnist_784_csv.csv");

        RecordReader rr = new CSVRecordReader(1, ',');
        rr.initialize(new FileSplit(f));

        //loop data in
        DataSetIterator iterator = new RecordReaderDataSetIterator(rr, batchSize, labelIndex, numClasses);
        /*List<List<Writable>> loop = new ArrayList<>();
        while (rr.hasNext())
        {
            loop.add(rr.next());
        }*/
        DataSet allData = iterator.next();

       /* // define schema
        Schema sch = new Schema.Builder()
                for (int i=0; i < 784; i++)
                {
                    .addColumn
                }
                .build();

        System.out.println(sch);

        System.out.println("no. of cols: " + sch.numColumns());
        System.out.println("col names: " + sch.getColumnNames());
        System.out.println("col types: " + sch.getColumnTypes());


        //split test-train
        //manipulate training data
        INDArray features = allData.getFeatures();
        features.*/

        List<List<Writable>> dataCollection = RecordConverter.toRecords(allData);
        INDArray dataArray = RecordConverter.toMatrix(DataType.FLOAT, dataCollection);



        //manipulate data
        allData.shuffle();
        //split for test-train
        SplitTestAndTrain test_train = allData.splitTestAndTrain(0.8);
        DataSet train = test_train.getTrain();
        DataSet test = test_train.getTest();
        DataSetIterator train_iter = new ViewIterator(train, batchSize);
        DataSetIterator test_iter = new ViewIterator(test, batchSize);

        //preprocessing
        NormalizerMinMaxScaler scaler = new NormalizerMinMaxScaler(0, 1);
        scaler.fit(train_iter);  //fit the scaler to training data
        //apply for both datasets
        train_iter.setPreProcessor(scaler);
        test_iter.setPreProcessor(scaler);
        System.out.println("input column = " + train_iter.inputColumns());

        //setting model config
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam())  //optimizer
                .weightInit(WeightInit.XAVIER)  //weight initialization
                .activation(Activation.RELU)  //default activation function
                .list()
                .layer(new DenseLayer.Builder()  //input layer
                        .nIn(train_iter.inputColumns())  //specify the no. of examples in training data
                        .nOut(124)
                        .build())
                .layer(new DenseLayer.Builder()  //hidden layer 1
                        .nOut(282)
                        .build())
                .layer(new OutputLayer.Builder()  //output layer
                        .lossFunction(LossFunctions.LossFunction.MCXENT)  //multiclass cross-entropy loss
                        .activation(Activation.SOFTMAX)
                        .nOut(10)  //specifiy the class nunmber for datasets
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
        int epoch = 10;
        for (int i = 0; i < epoch; ++i)
        {
            model.fit(train_iter);
            Evaluation evalTrain = model.evaluate(train_iter);  //validation with training data
            Evaluation evalTest = model.evaluate(test_iter);  //test with testing data
            System.out.println("\nepoch = " + i);
            System.out.println("Training accuracy: " + evalTrain.accuracy());
            System.out.println("Test  accuracy: " + evalTest.accuracy());

        }

        Evaluation evalTrain = model.evaluate(train_iter);  //validation with training data
        Evaluation evalTest = model.evaluate(test_iter);  //test with testing data
        System.out.println("Training: " + evalTrain.stats());
        System.out.println("Testing: " + evalTest.stats());


        System.out.println(evalTest.confusionMatrix());
    }
}
