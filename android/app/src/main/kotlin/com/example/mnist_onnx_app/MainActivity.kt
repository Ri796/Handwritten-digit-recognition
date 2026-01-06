package com.example.mnist_onnx_app

import android.os.Bundle
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {

    private val CHANNEL = "onnx_digit_classifier"
    private lateinit var classifier: OnnxDigitClassifier

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        classifier = OnnxDigitClassifier(this)

        MethodChannel(
            flutterEngine.dartExecutor.binaryMessenger,
            CHANNEL
        ).setMethodCallHandler { call, result ->

            if (call.method == "predict") {
                val input = call.argument<List<Double>>("input")

                if (input == null || input.size != 28 * 28) {
                    result.error("INVALID_INPUT", "Input must be 28x28", null)
                    return@setMethodCallHandler
                }

                val floatInput = FloatArray(input.size) {
                    input[it].toFloat()
                }

                val digit = classifier.predict(floatInput)
                result.success(digit)
            } else {
                result.notImplemented()
            }
        }
    }
}
