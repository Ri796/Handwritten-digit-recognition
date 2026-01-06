package com.example.mnist_onnx_app

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import android.content.Context
import java.nio.FloatBuffer

class OnnxDigitClassifier(private val context: Context) {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val modelBytes = context.assets.open("mnist_cnn.onnx").readBytes()
        session = env.createSession(modelBytes)
    }

    fun predict(input: FloatArray): Int {

        val shape = longArrayOf(1, 1, 28, 28)
        val buffer = FloatBuffer.wrap(input)

        val inputTensor = OnnxTensor.createTensor(
            env,
            buffer,
            shape
        )

        val results = session.run(
            mapOf(session.inputNames.first() to inputTensor)
        )

        val output = results[0].value as Array<FloatArray>
        val probs = output[0]

        return probs.indices.maxBy { probs[it] } ?: 0
    }
}
