# Keep ONNX Runtime classes
-keep class ai.onnxruntime.** { *; }

# Keep Native methods
-keepclasseswithmembernames class * {
    native <methods>;
}

# Keep the JNI interface
-keep class com.example.mnist_onnx_app.MainActivity { *; }
-keep class com.example.mnist_onnx_app.OnnxDigitClassifier { *; }
