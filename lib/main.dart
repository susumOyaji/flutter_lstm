//import 'package:flutter/material.dart';
//import 'package:stockcard_ai_pytorch/pytorch.dart';
//import 'package:stockcard_ai_pytorch/tesing_flutter_pytorch.dart';
//import 'package:stockcard_ai_pytorch/pytorch_mobile 0.2.1.dart';

import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';

import 'package:pytorch_mobile/pytorch_mobile.dart';
import 'package:pytorch_mobile/model.dart';
import 'package:pytorch_mobile/enums/dtype.dart';
import 'package:flutter_lstm/pytorch.dart';
import 'pytorch.dart';

void main() {
  runApp(const Mobile());
}

//void main() => runApp(Mobile());

class Tesing extends StatelessWidget {
  const Tesing({Key? key}) : super(key: key);
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: const TesingHomePage(),
    );
  }
}

class TesingHomePage extends StatefulWidget {
  const TesingHomePage({Key? key}) : super(key: key);

  @override
  _TesingHomePageState createState() => _TesingHomePageState();
}

class _TesingHomePageState extends State<TesingHomePage> {
  static const MethodChannel pytorchChannel =
      MethodChannel('com.pytorch_channel');

  @override
  void initState() {
    super.initState();
    _gettingModelFile().then((void value) => print('File Created Successfuly'));
  }

  String? documentsPath;
  String? prediction;

  Future<void> _gettingModelFile() async {
    final Directory directory = await getApplicationDocumentsDirectory();

    setState(() {
      documentsPath = directory.path;
    });

    final String resnet50 = join(directory.path, 'model.pt');
    final ByteData data = await rootBundle.load('assets/models/model.pt');

    final String segmodel = join(directory.path, 'seg_opt.pt');
    final ByteData segdata = await rootBundle.load('assets/models/seg_opt.pt');

    final List<int> bytes =
        data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
    final List<int> segbytes = segdata.buffer
        .asUint8List(segdata.offsetInBytes, segdata.lengthInBytes);

    if (!File(resnet50).existsSync()) {
      await File(resnet50).writeAsBytes(bytes);
    }
    if (!File(segmodel).existsSync()) {
      await File(segmodel).writeAsBytes(segbytes);
    }
  }

  Future<void> _getPrediction() async {
    final ByteData imageData = await rootBundle.load('assets/animal.jpg');
    try {
      final String result = await pytorchChannel.invokeMethod(
        'predict_image',
        <String, dynamic>{
          'model_path': '$documentsPath/model.pt',
          'image_data': imageData.buffer
              .asUint8List(imageData.offsetInBytes, imageData.lengthInBytes),
          'data_offset': imageData.offsetInBytes,
          'data_length': imageData.lengthInBytes
        },
      );
      setState(() {
        prediction = result;
      });
    } on PlatformException catch (e) {
      print(e);
    }
  }

  Future<void> _getSegmentation() async {
    final ByteData imageData = await rootBundle.load('assets/animal.jpg');
    try {
      final Float64List result = await pytorchChannel.invokeMethod(
        'segment_image',
        <String, dynamic>{
          'model_path': '$documentsPath/seg_opt.pt',
          'image_data': imageData.buffer
              .asUint8List(imageData.offsetInBytes, imageData.lengthInBytes),
          'data_offset': imageData.offsetInBytes,
          'data_length': imageData.lengthInBytes
        },
      );
      print(result);

      // setState(() {
      //   prediction = result;
      // });
    } on PlatformException catch (e) {
      print(e);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Pytorch Mobile'),
      ),
      body: Center(
        child: Column(
          children: <Widget>[
            Text(
              documentsPath ?? '',
              style: const TextStyle(fontSize: 8),
            ),
            Stack(
              children: <Widget>[
                Align(
                    alignment: Alignment.center,
                    child: SizedBox(
                        width: 300, child: Image.asset('assets/car.jpg'))),
                Align(
                  alignment: Alignment.bottomCenter,
                  child: Text(
                    (prediction ?? '').toUpperCase(),
                    style: const TextStyle(fontSize: 25),
                  ),
                ),
              ],
            ),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: <Widget>[
                ElevatedButton(
                  child: const Text('Segment Image'),
                  onPressed: _getSegmentation,
                ),
                ElevatedButton(
                  child: const Text('Classify Image'),
                  onPressed: _getPrediction,
                )
              ],
            )
          ],
        ),
      ),
      // floatingActionButton: FloatingActionButton(
      //   onPressed: _getPrediction,
      //   tooltip: 'Predict Image',
      //   child: const Icon(Icons.add),
      // ),
    );
  }
}

class Mobile extends StatefulWidget {
  const Mobile({Key? key}) : super(key: key);
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<Mobile> {
  Model? _imageModel, _customModel;

  String? _imagePrediction;
  List? _prediction;
  File? _image;
  ImagePicker _picker = ImagePicker();

  @override
  void initState() {
    super.initState();
    loadModel();
  }

  //load your model
  Future loadModel() async {
    String pathImageModel = "assets/models/resnet.pt";
    String pathCustomModel = "assets/models/custom_model.pt";
    try {
      _imageModel = await PyTorchMobile.loadModel(pathImageModel);
      _customModel = await PyTorchMobile.loadModel(pathCustomModel);
    } on PlatformException {
      print("only supported for android and ios so far");
    }
  }

  //run an image model
  Future runImageModel() async {
    //pick a random image
    final PickedFile? image = await _picker.getImage(
        source: (Platform.isIOS ? ImageSource.gallery : ImageSource.camera),
        maxHeight: 224,
        maxWidth: 224);
    //get prediction
    //labels are 1000 random english words for show purposes
    _imagePrediction = await _imageModel!.getImagePrediction(
        File(image!.path), 224, 224, "assets/labels/labels.csv");

    setState(() {
      _image = File(image.path);
    });
  }

  //run a custom model with number inputs
  Future runCustomModel() async {
    _prediction = await _customModel!
        .getPrediction([1, 2, 3, 4], [1, 2, 2], DType.float32);

    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: const Text('Pytorch Mobile Example(AI.Edition)'),
        ),
        body: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            _image == null ? Text('No image selected.') : Image.file(_image!),
            Center(
              child: Visibility(
                visible: _imagePrediction != null,
                child: Text("$_imagePrediction"),
              ),
            ),
            Center(
              child: TextButton(
                onPressed: runImageModel,
                child: Icon(
                  Icons.add_a_photo,
                  color: Colors.grey,
                ),
              ),
            ),
            TextButton(
              onPressed: runCustomModel,
              style: TextButton.styleFrom(
                backgroundColor: Colors.blue,
              ),
              child: Text(
                "Run custom model",
                style: TextStyle(
                  color: Colors.white,
                ),
              ),
            ),
            Center(
              child: Visibility(
                visible: _prediction != null,
                child: Text(_prediction != null ? "${_prediction![0]}" : ""),
              ),
            )
          ],
        ),
      ),
    );
  }
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        // This is the theme of your application.
        //
        // Try running your application with "flutter run". You'll see the
        // application has a blue toolbar. Then, without quitting the app, try
        // changing the primarySwatch below to Colors.green and then invoke
        // "hot reload" (press "r" in the console where you ran "flutter run",
        // or simply save your changes to "hot reload" in a Flutter IDE).
        // Notice that the counter didn't reset back to zero; the application
        // is not restarted.
        primarySwatch: Colors.blue,
      ),
      home: const MyHomePage(title: 'Flutter Demo Home Page'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({Key? key, required this.title}) : super(key: key);

  // This widget is the home page of your application. It is stateful, meaning
  // that it has a State object (defined below) that contains fields that affect
  // how it looks.

  // This class is the configuration for the state. It holds the values (in this
  // case the title) provided by the parent (in this case the App widget) and
  // used by the build method of the State. Fields in a Widget subclass are
  // always marked "final".

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  int _counter = 0;

  void _incrementCounter() {
    setState(() {
      // This call to setState tells the Flutter framework that something has
      // changed in this State, which causes it to rerun the build method below
      // so that the display can reflect the updated values. If we changed
      // _counter without calling setState(), then the build method would not be
      // called again, and so nothing would appear to happen.
      _counter++;
    });
  }

  @override
  Widget build(BuildContext context) {
    // This method is rerun every time setState is called, for instance as done
    // by the _incrementCounter method above.
    //
    // The Flutter framework has been optimized to make rerunning build methods
    // fast, so that you can just rebuild anything that needs updating rather
    // than having to individually change instances of widgets.
    return Scaffold(
      appBar: AppBar(
        // Here we take the value from the MyHomePage object that was created by
        // the App.build method, and use it to set our appbar title.
        title: Text(widget.title),
      ),
      body: Center(
        // Center is a layout widget. It takes a single child and positions it
        // in the middle of the parent.
        child: Column(
          // Column is also a layout widget. It takes a list of children and
          // arranges them vertically. By default, it sizes itself to fit its
          // children horizontally, and tries to be as tall as its parent.
          //
          // Invoke "debug painting" (press "p" in the console, choose the
          // "Toggle Debug Paint" action from the Flutter Inspector in Android
          // Studio, or the "Toggle Debug Paint" command in Visual Studio Code)
          // to see the wireframe for each widget.
          //
          // Column has various properties to control how it sizes itself and
          // how it positions its children. Here we use mainAxisAlignment to
          // center the children vertically; the main axis here is the vertical
          // axis because Columns are vertical (the cross axis would be
          // horizontal).
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            const Text(
              'You have pushed the button this many times:',
            ),
            Text(
              '$_counter',
              style: Theme.of(context).textTheme.headline4,
            ),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _incrementCounter,
        tooltip: 'Increment',
        child: const Icon(Icons.add),
      ), // This trailing comma makes auto-formatting nicer for build methods.
    );
  }
}
