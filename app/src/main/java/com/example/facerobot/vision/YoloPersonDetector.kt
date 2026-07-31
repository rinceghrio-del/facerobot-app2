package com.example.facerobot.vision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min

/**
 * Simpleng wrapper para sa isang YOLOv8-style TFLite model (anchor-free, output shape
 * [1, 4 + numClasses, numBoxes]) na ginagamit lang para tignan kung MAY TAO ba sa frame.
 *
 * PAALALA IDOL: Hindi kasama dito ang aktwal na model file. Kailangan mong maglagay ng
 * ".tflite" file sa: app/src/main/assets/yolo_person.tflite
 *
 * Saan kukuha ng model:
 *  - I-export mula sa Ultralytics YOLOv8n (COCO, 80 classes, "person" = class 0):
 *      pip install ultralytics
 *      yolo export model=yolov8n.pt format=tflite imgsz=320
 *    Lalabas na yolov8n_saved_model/yolov8n_float16.tflite - palitan pangalan/ilagay sa assets.
 *  - O kaya maghanap ng ready-made "yolov8n-fp16.tflite" sa GitHub/HuggingFace.
 *  - Kung gusto mo mas magaan, pwede mag-export ng int8 quantized version.
 *
 * Kung custom-trained ang model mo (person-only, 1 class), i-adjust ang PERSON_CLASS_INDEX
 * at NUM_CLASSES sa baba.
 */
class YoloPersonDetector(context: Context, modelAssetName: String = "yolo_person.tflite") {

    companion object {
        private const val TAG = "YoloPersonDetector"
        const val INPUT_SIZE = 320          // dapat tugma sa imgsz na ginamit sa export
        const val NUM_CLASSES = 80          // COCO default; baguhin kung custom-trained
        const val PERSON_CLASS_INDEX = 0    // "person" ang class 0 sa COCO
        const val CONF_THRESHOLD = 0.5f
        const val IOU_THRESHOLD = 0.45f
    }

    data class Detection(val box: RectF, val confidence: Float)

    private var interpreter: Interpreter? = null
    val isReady: Boolean get() = interpreter != null

    init {
        try {
            val model = loadModelFile(context, modelAssetName)
            val options = Interpreter.Options().apply { setNumThreads(4) }
            interpreter = Interpreter(model, options)
            Log.i(TAG, "YOLO model loaded: $modelAssetName")
        } catch (e: Exception) {
            // Sadyang hindi natin ipapa-crash ang app kung wala pa/mali ang model file -
            // gagana pa rin ang RoboEyes, wala lang auto person-detection.
            Log.e(TAG, "Hindi na-load ang YOLO model ($modelAssetName). Ilagay ito sa assets/.", e)
            interpreter = null
        }
    }

    private fun loadModelFile(context: Context, assetName: String): MappedByteBuffer {
        val afd = context.assets.openFd(assetName)
        FileInputStream(afd.fileDescriptor).use { input ->
            val channel = input.channel
            return channel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
        }
    }

    /** Nagbabalik ng listahan ng "person" detections. Empty list kung wala o kung walang model. */
    fun detectPersons(bitmap: Bitmap): List<Detection> {
        val interp = interpreter ?: return emptyList()

        val resized = ImageUtils.resize(bitmap, INPUT_SIZE)
        val inputBuffer = bitmapToInputBuffer(resized)

        val outputShape = interp.getOutputTensor(0).shape() // e.g. [1, 84, 2100]
        val output = Array(outputShape[0]) { Array(outputShape[1]) { FloatArray(outputShape[2]) } }

        try {
            interp.run(inputBuffer, output)
        } catch (e: Exception) {
            Log.e(TAG, "YOLO inference failed", e)
            return emptyList()
        }

        return decodeOutput(output[0], outputShape, bitmap.width, bitmap.height)
    }

    private fun bitmapToInputBuffer(bitmap: Bitmap): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3)
        buffer.order(ByteOrder.nativeOrder())
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        for (pixel in pixels) {
            buffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f) // R
            buffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)  // G
            buffer.putFloat((pixel and 0xFF) / 255.0f)          // B
        }
        buffer.rewind()
        return buffer
    }

    /**
     * Output layout na inaasahan: [numAttributes][numBoxes] kung saan numAttributes = 4 + NUM_CLASSES
     * (cx, cy, w, h, tapos yung class scores). Ito ang karaniwang layout ng YOLOv8 tflite export
     * na naka-transpose na. Kung iba ang shape ng model mo, i-adjust ito.
     */
    private fun decodeOutput(
        output: Array<FloatArray>,
        shape: IntArray,
        origWidth: Int,
        origHeight: Int
    ): List<Detection> {
        val numAttrs = shape[1]
        val numBoxes = shape[2]
        if (numAttrs < 4 + PERSON_CLASS_INDEX + 1) return emptyList()

        val candidates = mutableListOf<Detection>()

        for (i in 0 until numBoxes) {
            val classScore = output[4 + PERSON_CLASS_INDEX][i]
            if (classScore < CONF_THRESHOLD) continue

            val cx = output[0][i] / INPUT_SIZE * origWidth
            val cy = output[1][i] / INPUT_SIZE * origHeight
            val w = output[2][i] / INPUT_SIZE * origWidth
            val h = output[3][i] / INPUT_SIZE * origHeight

            val rect = RectF(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            candidates.add(Detection(rect, classScore))
        }

        return nonMaxSuppression(candidates)
    }

    private fun nonMaxSuppression(detections: List<Detection>): List<Detection> {
        val sorted = detections.sortedByDescending { it.confidence }.toMutableList()
        val result = mutableListOf<Detection>()

        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            result.add(best)
            sorted.removeAll { iou(best.box, it.box) > IOU_THRESHOLD }
        }
        return result
    }

    private fun iou(a: RectF, b: RectF): Float {
        val left = max(a.left, b.left)
        val top = max(a.top, b.top)
        val right = min(a.right, b.right)
        val bottom = min(a.bottom, b.bottom)

        val intersection = max(0f, right - left) * max(0f, bottom - top)
        val areaA = (a.right - a.left) * (a.bottom - a.top)
        val areaB = (b.right - b.left) * (b.bottom - b.top)
        val union = areaA + areaB - intersection
        return if (union <= 0f) 0f else intersection / union
    }

    fun close() {
        interpreter?.close()
        interpreter = null
    }
}
