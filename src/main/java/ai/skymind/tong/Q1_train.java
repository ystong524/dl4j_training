package ai.skymind.tong;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.filter.FilterInvalidValues;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.MinMaxSerializerStrategy;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

public class Q1_train {
    private static File inputFile;
    public static void main(String[] args) throws Exception {
        //Path f1 = Paths.get();
        File inputFile = new File("C:/Users/user/Desktop/Penjana_DLPC/Day11/Q1/train.csv");

        // read csv file
        FileSplit fs = new FileSplit(inputFile);
        RecordReader rr = new CSVRecordReader(1, ',');
        rr.initialize(fs);

        /*//ID,age,job,marital,education,default,balance,housing,loan,contact,day,month,duration,campaign,pdays,previous,poutcome,subscribed
        // define schema
        Schema sch = new Schema.Builder()
                .addColumnInteger("ID")
                .addColumnInteger("age")
                .addColumnCategorical("job", Arrays.asList("admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"))
                .addColumnCategorical("marital", Arrays.asList("divorced", "married", "single"))
                .addColumnCategorical("education", Arrays.asList("primary", "secondary", "tertiary", "unknown"))
                .addColumnCategorical("default", Arrays.asList("no", "yes"))
                .addColumnInteger("balance")
                .addColumnCategorical("housing", Arrays.asList("no", "yes"))
                .addColumnCategorical("loan", Arrays.asList("no", "yes"))
                .addColumnCategorical("contact", Arrays.asList("cellular", "telephone", "unknown"))
                .addColumnInteger("day", 1, 31)
                .addColumnCategorical("month", Arrays.asList("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"))
                .addColumnInteger("duration")
                .addColumnInteger("campaign")
                .addColumnInteger("pdays")
                .addColumnInteger("previous")
                .addColumnCategorical("poutcome", Arrays.asList("failure", "other", "success", "unknown"))
                .addColumnCategorical("subscribed", Arrays.asList("no", "yes"))
                .build();*/

        Schema sch = getTrainSchema();
        System.out.println("Original data");
        System.out.println(sch);

        System.out.println("no. of cols: " + sch.numColumns());
        System.out.println("col names: " + sch.getColumnNames());
        System.out.println("col types: " + sch.getColumnTypes());

        //loop from rr
        List<List<Writable>> oriData = new ArrayList<>();
        while (rr.hasNext())
        {
            oriData.add(rr.next());
        }

        /*//perform pca
        sch.getIndexOfColumn("subscribed");
        PCA pca = new PCA(data);*/

        TransformProcess tp = new TransformProcess.Builder(sch)
                .categoricalToInteger("job", "marital", "education", "default", "housing",
                        "loan", "contact", "month", "poutcome", "subscribed")
                .filter(new FilterInvalidValues())
                .build();

        /*//ID,age,job,marital,education,default,balance,housing,loan,contact,day,month,duration,campaign,pdays,previous,poutcome,subscribed
        TransformProcess tp = new TransformProcess.Builder(sch)
                .removeColumns("ID", "day", "month")// "duration", "pdays", "campaign", "contact", "poutcome", "default", "previous")  //ignore id and call time
                .categoricalToOneHot("job")
                .categoricalToOneHot("marital")
                .categoricalToOneHot("education")  //ordinal, but unknown??
                .categoricalToInteger("default")
                .categoricalToInteger("housing")
                .categoricalToInteger("loan")
                .categoricalToOneHot("contact")  //non-ordinal
                .categoricalToOneHot("poutcome")  //non-ordinal

                //one hot the target
                .categoricalToInteger("subscribed")  //binary class
                //filter invalid
                .filter(new FilterInvalidValues())
                .build();*/

        Schema final_sch = tp.getFinalSchema();
        System.out.println("After transformation");
        System.out.println(sch);

        System.out.println("no. of cols: " + final_sch.numColumns());
        System.out.println("col names: " + final_sch.getColumnNames());
        System.out.println("col types: " + final_sch.getColumnTypes());


        //apply transform
        List<List<Writable>> transData = LocalTransformExecutor.execute(oriData, tp);
        RecordReader crr = new CollectionRecordReader(transData);
        int labelIndex = sch.getIndexOfColumn("subscribed");
        int possible = 2;
        int batchSize = 2000;
        System.out.println("label at " + labelIndex);

        List<String> label_names = final_sch.getColumnNames();


        DataSetIterator iter = new RecordReaderDataSetIterator(crr, transData.size(), labelIndex, possible);
        DataSet allData = iter.next();

        //manipulate training data
        allData.shuffle();
        //split for validation
        System.out.println("allData.numInputs() = " + allData.numInputs());
        System.out.println("allData.numOutcomes() = " + allData.numOutcomes());

        //total == 31647 data
        SplitTestAndTrain test_train = allData.splitTestAndTrain(0.8);
        DataSet train = test_train.getTrain();
        DataSet val = test_train.getTest();

        //normalization before training
        DataNormalization scaler = new NormalizerMinMaxScaler(0, 1);


        /*//1. apply scaler to DataSet
        scaler.fit(train);
        scaler.transform(train);
        scaler.transform(val);
        //end 1.*/

        System.out.println("train.numExamples() = " + train.numExamples());
        System.out.println("val.numExamples() = " + val.numExamples());

        DataSetIterator train_iter = new ViewIterator(train, batchSize);
        DataSetIterator val_iter = new ViewIterator(val,batchSize);

        //2. set preprocessor to iterator
        scaler.fit(train_iter);
        train_iter.setPreProcessor(scaler);
        val_iter.setPreProcessor(scaler);
        //end 2.

        System.out.println("input column = " + train_iter.inputColumns());

        HashMap<Integer, Double> schedule = new HashMap<>();
        schedule.put(0, 1e-3);
        schedule.put(4, 1e-4);
        schedule.put(6, 1e-5);

        //setup NN
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(1234)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(new MapSchedule(ScheduleType.EPOCH, schedule)))
                //.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                //.miniBatch(true)
                .l2(0.0001)
                //.dropOut(0.9)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(train_iter.inputColumns())
                        .nOut(300)
                        .activation(Activation.RELU)
                        .build())
                .layer(new BatchNormalization())
                .layer(new DenseLayer.Builder()
                        .nOut(500)
                        .activation(Activation.RELU)
                        .build())
                .layer(new BatchNormalization())
                .layer(new DropoutLayer.Builder()
                    .nOut(200)
                    //.dropOut(0.6)
                    .activation(Activation.RELU)
                    .build())
                .layer(new BatchNormalization())

                /*.layer(new BatchNormalization())
                .layer( new DenseLayer.Builder()
                        .nOut(200)
                        .activation(Activation.RELU)
                        .build())
                .layer(new BatchNormalization())
                .layer( new DenseLayer.Builder()
                        .nOut(200)
                        .activation(Activation.RELU)
                        .build())*/
                .layer(new OutputLayer.Builder()
                        .activation(Activation.SIGMOID)
                        .lossFunction(LossFunctions.LossFunction.XENT)
                        .nOut(train_iter.totalOutcomes())
                        .build())
                .build();

        //setup model
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        InMemoryStatsStorage storage = new InMemoryStatsStorage(); //allocate some memory
        UIServer server = UIServer.getInstance();
        server.attach(storage);  //attach memory to use for server
        model.setListeners(new StatsListener(storage), new ScoreIterationListener(1000));

        //training model
        int epoch = 10;
        for (int i = 0; i < epoch; ++i)
        {
            model.fit(train_iter);
            Evaluation evalTrain = model.evaluate(train_iter);  //validation with training data
            Evaluation evalTest = model.evaluate(val_iter);  //test with testing data
            System.out.println("\nepoch = " + i);
            System.out.println("Training: " + evalTrain.stats());
            System.out.println("Validation: " + evalTest.stats());

        }

        /*Evaluation evalTrain = model.evaluate(train_iter);  //validation with training data
        Evaluation evalTest = model.evaluate(val_iter);  //test with testing data
        System.out.println("Training: " + evalTrain.stats());
        System.out.println("Validation: " + evalTest.stats());
        //INDArray predict = model.output(test_iter);*/

        //Save model
        ModelSerializer.writeModel(model, "C:/Users/user/Desktop/Penjana_DLPC/Day11/Q1/model.zip", true);
        NormalizerSerializer normalizerSerializer = new NormalizerSerializer().addStrategy(new MinMaxSerializerStrategy());
        normalizerSerializer.write(scaler, "C:/Users/user/Desktop/Penjana_DLPC/Day11/Q1/normalizer.zip");

        //Testing
        Nd4j.getEnvironment().allowHelpers(false); //required for CPU - concat error
        //Convert val dataset to INDArray
        List<List<Writable>> valCollection = RecordConverter.toRecords(val);
        INDArray valArray = RecordConverter.toMatrix(DataType.FLOAT, valCollection);
        INDArray valFeatures = valArray.getColumns(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16);
        //INDArray valFeatures = valArray.getColumns(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,);


        train.setLabelNames(Arrays.asList("0", "1"));
        val.setLabelNames(Arrays.asList("0", "1"));
        List<String> prediction = model.predict(val);
        INDArray output = model.output(valFeatures);

        //System.out.println(prediction);
        //System.out.println(output);

        for (int i=0; i < 10; i++)
        {
            System.out.println("Prediction: " + prediction.get(i) + "; Output: " + output.getRow(i));
        }

    }

    static Schema getTrainSchema() {

        return new Schema.Builder()
                .addColumnsInteger("ID", "age")
                .addColumnCategorical("job",
                        Arrays.asList("admin.", "blue-collar", "entrepreneur", "housemaid", "management",
                                "retired", "self-employed", "services", "student", "technician",
                                "unemployed", "unknown"))
                .addColumnCategorical("marital", Arrays.asList("married", "divorced", "single"))
                .addColumnCategorical("education", Arrays.asList("unknown", "secondary", "tertiary", "primary"))
                .addColumnCategorical("default", Arrays.asList("no", "yes"))
                .addColumnDouble("balance")
                .addColumnCategorical("housing", Arrays.asList("no", "yes"))
                .addColumnCategorical("loan", Arrays.asList("no", "yes"))
                .addColumnCategorical("contact", Arrays.asList("telephone", "cellular", "unknown"))
                .addColumnInteger("day")
                .addColumnCategorical("month", Arrays.asList("jan", "feb", "mar", "apr", "may", "jun",
                        "jul", "aug", "sep", "oct", "nov", "dec"))
                .addColumnsInteger("duration", "campaign", "pdays", "previous")
                .addColumnCategorical("poutcome", Arrays.asList("unknown", "success", "failure", "other"))
                .addColumnCategorical("subscribed", Arrays.asList("no", "yes"))
                .build();
    }

}
