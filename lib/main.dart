import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img; // 'image' package to handle resizing/pixels
import 'dart:io';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(primarySwatch: Colors.indigo),
      home: const DigitRecognizerScreen(),
    );
  }
}

class DigitRecognizerScreen extends StatefulWidget {
  const DigitRecognizerScreen({super.key});

  @override
  State<DigitRecognizerScreen> createState() => _DigitRecognizerScreenState();
}

class _DigitRecognizerScreenState extends State<DigitRecognizerScreen> with SingleTickerProviderStateMixin {
  // Tabs
  late TabController _tabController;

  // Drawing State
  List<Offset?> _points = [];
  
  // Upload State
  File? _selectedImage;
  
  // Prediction result
  String _prediction = "Draw or Upload a digit!";
  
  // Platform channel
  static const platform = MethodChannel('onnx_digit_classifier');

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 2, vsync: this);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Digit Recognizer'),
        backgroundColor: Colors.indigo,
        foregroundColor: Colors.white,
        bottom: TabBar(
          controller: _tabController,
          labelColor: Colors.white,
          unselectedLabelColor: Colors.white70,
          indicatorColor: Colors.white,
          tabs: const [
            Tab(icon: Icon(Icons.gesture), text: "Draw"),
            Tab(icon: Icon(Icons.upload), text: "Upload"),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabController,
        children: [
          _buildDrawingTab(),
          _buildUploadTab(),
        ],
      ),
      bottomNavigationBar: Container(
        padding: const EdgeInsets.all(20),
        color: Colors.grey[200],
        child: Text(
          _prediction,
          textAlign: TextAlign.center,
          style: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold, color: Colors.indigo),
        ),
      ),
    );
  }

  // --- Drawing Tab ---
  Widget _buildDrawingTab() {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Container(
          width: 280,
          height: 280,
          decoration: BoxDecoration(
            border: Border.all(color: Colors.black, width: 2),
            color: Colors.black, 
          ),
          child: GestureDetector(
            onPanUpdate: (details) {
              setState(() {
                RenderBox renderBox = context.findRenderObject() as RenderBox;
                // Need to locate the Container specifically? 
                // Actually typically globalToLocal on the widget's context works if the widget captures the gesture.
                // But finding renderObject of the *Screen* might be offset.
                // Let's use a local key or context?
                // Simplest way: use the local position from 'details' if available with GestureDetector context.
                // However, GestureDetector doesn't give context easily in update.
                // Let's rely on global.
                // Correction: The safest way is using a Builder to get context of the container, 
                // but let's assume centered layout allows simple offset logic or just use Overlay. 
                // Actually GestureDetector exposes 'localPosition' too!
                _points.add(details.localPosition);
              });
            },
            onPanEnd: (details) {
              _points.add(null);
              _predictDrawing(); 
            },
            child: CustomPaint(
              size: const Size(280, 280),
              painter: DrawingPainter(_points),
            ),
          ),
        ),
        const SizedBox(height: 20),
        ElevatedButton.icon(
          onPressed: () {
            setState(() {
              _points.clear();
              _prediction = "Draw a digit!";
            });
          },
          icon: const Icon(Icons.clear),
          label: const Text('Clear Canvas'),
        ),
      ],
    );
  }

  // --- Upload Tab ---
  Widget _buildUploadTab() {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Container(
          width: 280,
          height: 280,
          decoration: BoxDecoration(
            border: Border.all(color: Colors.grey),
            color: Colors.grey[300],
          ),
          child: _selectedImage == null
              ? const Center(child: Icon(Icons.image, size: 50, color: Colors.grey))
              : Image.file(_selectedImage!, fit: BoxFit.contain),
        ),
        const SizedBox(height: 20),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton.icon(
              onPressed: () => _pickImage(ImageSource.gallery),
              icon: const Icon(Icons.photo_library),
              label: const Text('Gallery'),
            ),
            const SizedBox(width: 20),
            ElevatedButton.icon(
              onPressed: () => _pickImage(ImageSource.camera),
              icon: const Icon(Icons.camera_alt),
              label: const Text('Camera'),
            ),
          ],
        ),
      ],
    );
  }

  // --- Logic ---

  Future<void> _pickImage(ImageSource source) async {
    final picker = ImagePicker();
    final XFile? image = await picker.pickImage(source: source);
    
    if (image != null) {
      setState(() {
        _selectedImage = File(image.path);
        _prediction = "Processing...";
      });
      _predictUpload(File(image.path));
    }
  }

  Future<void> _predictUpload(File imageFile) async {
    try {
      final bytes = await imageFile.readAsBytes();
      final img.Image? originalImage = img.decodeImage(bytes);
      
      if (originalImage == null) return;

      // 1. Resize to 28x28
      // The model was trained on 28x28 pixel images, so we must resize whatever the user uploaded.
      final img.Image resized = img.copyResize(originalImage, width: 28, height: 28);

      // 2. Convert to Grayscale & Normalize
      // Our model expects (Value - Mean) / Std.
      // Mean = 0.1307, Std = 0.3081
      const mean = 0.1307;
      const std = 0.3081;
      List<double> normalizedInput = [];

      for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
          final pixel = resized.getPixel(x, y);
          
          // Get luminance (grayscale value 0-255)
          final lum = img.getLuminance(pixel);
          
          // INVERSION LOGIC:
          // Use Case A: User takes photo of black ink on white paper.
          // Model expects: White digit on black background (like standard MNIST).
          // Therefore, we assume we need to invert the colors (0 becomes 255, 255 becomes 0).
          // Normalized value range should be roughly -0.42 (black) to 2.8 (white).
          
          double invertedVal = (255 - lum).toDouble(); 
          
          // Mathematical normalization
          double normalized = (invertedVal / 255.0 - mean) / std;
          normalizedInput.add(normalized);
        }
      }

      await _sendToNative(normalizedInput);

    } catch (e) {
      setState(() => _prediction = "Error: $e");
    }
  }

  Future<void> _predictDrawing() async {
    // 1. Snapshot the canvas
    // We render the specific drawing strokes into a small 28x28 bitmap.
    final recorder = ui.PictureRecorder();
    final canvas = Canvas(recorder, Rect.fromPoints(const Offset(0, 0), const Offset(28, 28)));
    
    // We scale the canvas down by 0.1 because our drawing area is 280x280.
    canvas.scale(0.1, 0.1); 
    
    // Draw Black Background
    canvas.drawRect(const Rect.fromLTWH(0, 0, 280, 280), Paint()..color = Colors.black);
    
    // Draw White Strokes
    // We use a thick stroke (20.0) so it shows up clearly when shrunk to 28x28.
    final paint = Paint()
      ..color = Colors.white
      ..strokeCap = StrokeCap.round
      ..strokeWidth = 20.0; 

    for (int i = 0; i < _points.length - 1; i++) {
        if (_points[i] != null && _points[i + 1] != null) {
            canvas.drawLine(_points[i]!, _points[i + 1]!, paint);
        }
    }
    
    final picture = recorder.endRecording();
    final imgUi = await picture.toImage(28, 28);
    final byteData = await imgUi.toByteData(format: ui.ImageByteFormat.rawRgba);
    
    if (byteData == null) return;
    final bytes = byteData.buffer.asUint8List();

    // 2. Prepare for Model
    const mean = 0.1307;
    const std = 0.3081;
    List<double> normalizedInput = [];
    
    // Iterate pixels. Each pixel is 4 bytes (R, G, B, A)
    for (int i = 0; i < bytes.length; i += 4) {
      final r = bytes[i]; // Since we drew white on black, R=G=B=Luminance.
      
      double val = r / 255.0; 
      double normalized = (val - mean) / std;
      normalizedInput.add(normalized);
    }

    await _sendToNative(normalizedInput);
  }

  Future<void> _sendToNative(List<double> input) async {
    try {
      final int result = await platform.invokeMethod('predict', {'input': input});
      setState(() {
        _prediction = "Prediction: $result";
      });
    } on PlatformException catch (e) {
      setState(() {
        _prediction = "Native Error: '${e.message}'";
      });
    }
  }
}

class DrawingPainter extends CustomPainter {
  final List<Offset?> points;
  DrawingPainter(this.points);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.white
      ..strokeCap = StrokeCap.round
      ..strokeWidth = 10.0;

    for (int i = 0; i < points.length - 1; i++) {
      if (points[i] != null && points[i + 1] != null) {
        canvas.drawLine(points[i]!, points[i + 1]!, paint);
      }
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
