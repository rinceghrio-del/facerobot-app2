package com.example.facerobot

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Rect
import android.os.Bundle
import android.text.InputType
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.facerobot.ui.RoboEyesView
import com.example.facerobot.vision.FaceEmbedder
import com.example.facerobot.vision.FaceStore
import com.example.facerobot.vision.ImageUtils
import com.example.facerobot.vision.YoloPersonDetector
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import okhttp3.OkHttpClient
import okhttp3.Request
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * FaceRobot MainActivity - buong flow:
 *
 *   [RoboEyes idle screen] --(YOLO nakakita ng tao)--> [Camera + face recognition] --(walang
 *   tao ilang segundo)--> [balik sa RoboEyes]
 *
 * Habang nasa CAMERA state, patuloy na sinusundan ang mukha (LEFT/RIGHT/FORWARD/STOP command
 * papunta sa ESP32) at sinusubukang kilalanin kung sino gamit ang naka-enroll na mga mukha.
 */
@androidx.camera.core.ExperimentalGetImage
class MainActivity : ComponentActivity() {

    private enum class AppState { EYES, CAMERA }

    // ---------- UI (gawa lahat sa Kotlin code, walang XML layout) ----------
    private lateinit var rootLayout: FrameLayout
    private lateinit var roboEyesView: RoboEyesView
    private lateinit var previewView: PreviewView
    private lateinit var statusText: TextView
    private lateinit var enrollButton: Button

    private lateinit var cameraExecutor: ExecutorService
    private val httpClient = OkHttpClient()

    private lateinit var yoloDetector: YoloPersonDetector
    private lateinit var faceEmbedder: FaceEmbedder
    private lateinit var faceStore: FaceStore

    private var appState = AppState.EYES

    // PALITAN ITO ng IP address ng ESP32 mo (makikita sa Serial Monitor pag nag-boot)
    private val esp32BaseUrl = "http://192.168.4.1"

    // Throttle para sa command papunta sa ESP32
    private var lastSendTime = 0L
    private val sendIntervalMs = 300L

    // Throttle para sa YOLO (mabigat siya kaysa ML Kit, kaya bihira lang patakbuhin)
    private var lastYoloCheckTime = 0L
    private val yoloIntervalMs = 400L

    // Throttle para sa face recognition (embedding + compare)
    private var lastRecognitionTime = 0L
    private val recognitionIntervalMs = 600L

    // Kung gaano katagal walang nakikitang tao bago bumalik sa RoboEyes
    private var lastPersonSeenTime = 0L
    private val personTimeoutMs = 4000L

    // Huling nakuhang embedding ng "hindi kilalang" mukha - gagamitin ng enroll button
    private var lastUnknownFaceEmbedding: FloatArray? = null

    private val faceDetectorOptions = FaceDetectorOptions.Builder()
        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
        .build()
    private val faceDetector = FaceDetection.getClient(faceDetectorOptions)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        cameraExecutor = Executors.newSingleThreadExecutor()
        yoloDetector = YoloPersonDetector(this)
        faceEmbedder = FaceEmbedder(this)
        faceStore = FaceStore(this)

        buildUi()
        showEyesUi()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 100)
        }
    }

    // ---------- UI setup ----------

    private fun buildUi() {
        rootLayout = FrameLayout(this)
        roboEyesView = RoboEyesView(this)
        previewView = PreviewView(this)

        statusText = TextView(this).apply {
            setTextColor(0xFFFFFFFF.toInt())
            textSize = 16f
            setPadding(24, 24, 24, 24)
            setBackgroundColor(0x88000000.toInt())
            gravity = Gravity.CENTER
        }

        enrollButton = Button(this).apply {
            text = "Mag-enroll ng bagong mukha"
            visibility = View.GONE
            setOnClickListener { showEnrollDialog() }
        }

        rootLayout.addView(
            roboEyesView,
            FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT)
        )
        rootLayout.addView(
            previewView,
            FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT)
        )
        rootLayout.addView(
            statusText,
            FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.WRAP_CONTENT)
                .apply { gravity = Gravity.TOP }
        )
        rootLayout.addView(
            enrollButton,
            FrameLayout.LayoutParams(FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT)
                .apply { gravity = Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL; bottomMargin = 48 }
        )

        setContentView(rootLayout)
    }

    private fun showEyesUi() {
        appState = AppState.EYES
        roboEyesView.visibility = View.VISIBLE
        previewView.visibility = View.GONE
        enrollButton.visibility = View.GONE
        roboEyesView.setMood(RoboEyesView.Mood.SEARCHING)
        statusText.text = if (yoloDetector.isReady) {
            "Naghahanap ng tao..."
        } else {
            "Naghahanap ng tao... (kulang: assets/yolo_person.tflite)"
        }
    }

    private fun showCameraUi() {
        appState = AppState.CAMERA
        roboEyesView.visibility = View.GONE
        previewView.visibility = View.VISIBLE
        lastPersonSeenTime = System.currentTimeMillis()
        statusText.text = "May tao! Sinusubukang kilalanin..."
    }

    private fun runOnUi(block: () -> Unit) = runOnUiThread(block)

    // ---------- Camera setup (iisang Preview + ImageAnalysis lang, pinapalitan lang
    // yung ginagawa ng analyzer depende sa kung anong appState) ----------

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()

                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

                val imageAnalyzer = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also {
                        it.setAnalyzer(cameraExecutor) { imageProxy -> processFrame(imageProxy) }
                    }

                // Front camera - kasabay ng screen, kaya nakaharap din sa taong nakatingin sa RoboEyes
                val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)
            } catch (e: Exception) {
                e.printStackTrace()
                runOnUi {
                    statusText.text = "Naghahanap ng tao... (camera setup error: ${e.javaClass.simpleName}: ${e.message})"
                }
            }
        }, ContextCompat.getMainExecutor(this))
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 100 && grantResults.isNotEmpty()
            && grantResults[0] == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        }
    }

    // ---------- Frame processing ----------

    private fun processFrame(imageProxy: ImageProxy) {
        when (appState) {
            AppState.EYES -> processEyesFrame(imageProxy)
            AppState.CAMERA -> processCameraFrame(imageProxy)
        }
    }

    /** EYES mode: paminsan-minsan lang mag-run ng YOLO, tignan lang kung may tao. */
    private fun processEyesFrame(imageProxy: ImageProxy) {
        val now = System.currentTimeMillis()
        if (!yoloDetector.isReady || now - lastYoloCheckTime < yoloIntervalMs) {
            imageProxy.close()
            return
        }
        lastYoloCheckTime = now

        try {
            val bitmap = ImageUtils.imageProxyToBitmap(imageProxy)
            val detections = yoloDetector.detectPersons(bitmap)

            runOnUi {
                val err = yoloDetector.lastError
                statusText.text = when {
                    err != null -> "Naghahanap ng tao... (YOLO error: $err)"
                    else -> "Naghahanap ng tao... (confidence: %.2f)".format(yoloDetector.lastMaxPersonScore)
                }
            }

            if (detections.isNotEmpty()) {
                runOnUi { roboEyesView.setMood(RoboEyesView.Mood.ALERT) }
                // Konting delay para makita muna ang "alert" na expression bago lumipat
                rootLayout.postDelayed({
                    if (appState == AppState.EYES) showCameraUi()
                }, 350)
            }
        } catch (e: Exception) {
            e.printStackTrace()
            runOnUi {
                statusText.text = "Naghahanap ng tao... (crash: ${e.javaClass.simpleName}: ${e.message})"
            }
        } finally {
            imageProxy.close()
        }
    }

    /** CAMERA mode: face detection (tracking) + recognition (kilalanin) + ESP32 commands. */
    private fun processCameraFrame(imageProxy: ImageProxy) {
        val mediaImage = imageProxy.image
        if (mediaImage == null) {
            imageProxy.close()
            return
        }

        val rotation = imageProxy.imageInfo.rotationDegrees
        val inputImage = InputImage.fromMediaImage(mediaImage, rotation)

        faceDetector.process(inputImage)
            .addOnSuccessListener { faces ->
                if (faces.isNotEmpty()) {
                    handleFaceFound(faces[0], imageProxy, rotation)
                } else {
                    handleNoFace()
                }
            }
            .addOnFailureListener { it.printStackTrace() }
            .addOnCompleteListener { imageProxy.close() }
    }

    private fun handleFaceFound(face: Face, imageProxy: ImageProxy, rotation: Int) {
        lastPersonSeenTime = System.currentTimeMillis()

        val box = face.boundingBox
        val frameWidth = imageProxy.width
        val frameHeight = imageProxy.height

        // Sundan ang mukha (parehong logic ng orihinal na bersyon)
        val command = computeCommand(box.centerX(), box.width(), frameWidth)
        sendCommandThrottled(command)

        // Face recognition - bihira lang gawin (mabigat kung bawat frame)
        val now = System.currentTimeMillis()
        if (faceEmbedder.isReady && now - lastRecognitionTime > recognitionIntervalMs) {
            lastRecognitionTime = now
            try {
                val bitmap = ImageUtils.imageProxyToBitmap(imageProxy)
                val adjustedBox = adjustBoxForRotation(box, frameWidth, frameHeight, rotation)
                val faceCrop = ImageUtils.safeCrop(bitmap, adjustedBox)

                if (faceCrop != null) {
                    val embedding = faceEmbedder.getEmbedding(faceCrop)
                    if (embedding != null) {
                        val match = faceStore.match(embedding)
                        runOnUi {
                            if (match != null) {
                                statusText.text = "Kilala: ${match.name} (${(match.similarity * 100).toInt()}%)"
                                enrollButton.visibility = View.GONE
                                lastUnknownFaceEmbedding = null
                            } else {
                                statusText.text = if (faceStore.isEmpty()) {
                                    "May tao pero wala pang naka-enroll na mukha"
                                } else {
                                    "Hindi kilala"
                                }
                                lastUnknownFaceEmbedding = embedding
                                enrollButton.visibility = View.VISIBLE
                            }
                        }
                    }
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    private fun handleNoFace() {
        sendCommandThrottled("SEARCH")
        val now = System.currentTimeMillis()
        if (now - lastPersonSeenTime > personTimeoutMs) {
            runOnUi { showEyesUi() }
        }
    }

    /**
     * Ang Face.boundingBox mula sa ML Kit ay nasa coordinate space ng ORIHINAL (hindi pa
     * piniko) na frame, pero ang bitmap na galing sa ImageUtils.imageProxyToBitmap ay
     * naka-rotate na. Kailangan i-convert ang box papunta sa parehong "rotated" space.
     */
    private fun adjustBoxForRotation(box: Rect, frameWidth: Int, frameHeight: Int, rotationDegrees: Int): Rect {
        return when (rotationDegrees) {
            90 -> Rect(frameHeight - box.bottom, box.left, frameHeight - box.top, box.right)
            180 -> Rect(frameWidth - box.right, frameHeight - box.bottom, frameWidth - box.left, frameHeight - box.top)
            270 -> Rect(box.top, frameWidth - box.right, box.bottom, frameWidth - box.left)
            else -> box
        }
    }

    /**
     * Simpleng logic para malaman kung saang direksyon dapat gumalaw ang robot.
     */
    private fun computeCommand(faceCenterX: Int, faceWidth: Int, frameWidth: Int): String {
        val screenCenterX = frameWidth / 2
        val centerZoneWidth = frameWidth / 6
        val closeThreshold = frameWidth / 3

        return when {
            faceWidth > closeThreshold -> "STOP"
            faceCenterX < screenCenterX - centerZoneWidth -> "LEFT"
            faceCenterX > screenCenterX + centerZoneWidth -> "RIGHT"
            else -> "FORWARD"
        }
    }

    private fun sendCommandThrottled(command: String) {
        val now = System.currentTimeMillis()
        if (now - lastSendTime < sendIntervalMs) return
        lastSendTime = now
        sendCommandToEsp32(command)
    }

    private fun sendCommandToEsp32(command: String) {
        val request = Request.Builder()
            .url("$esp32BaseUrl/command?dir=$command")
            .build()

        httpClient.newCall(request).enqueue(object : okhttp3.Callback {
            override fun onFailure(call: okhttp3.Call, e: java.io.IOException) {
                // OK lang kung minsan mag-fail (weak WiFi signal, etc.)
            }
            override fun onResponse(call: okhttp3.Call, response: okhttp3.Response) {
                response.close()
            }
        })
    }

    // ---------- Enroll UI ----------

    private fun showEnrollDialog() {
        val embedding = lastUnknownFaceEmbedding
        if (embedding == null) {
            statusText.text = "Wala pang mukha na nakuha, subukan ulit"
            return
        }

        val input = EditText(this).apply {
            hint = "Pangalan (hal. Rusty)"
            inputType = InputType.TYPE_CLASS_TEXT
        }

        android.app.AlertDialog.Builder(this)
            .setTitle("Mag-enroll ng mukha")
            .setView(input)
            .setPositiveButton("Save") { _, _ ->
                val name = input.text.toString().trim()
                if (name.isNotEmpty()) {
                    faceStore.enroll(name, embedding)
                    statusText.text = "Na-enroll: $name"
                    enrollButton.visibility = View.GONE
                    lastUnknownFaceEmbedding = null
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        faceDetector.close()
        yoloDetector.close()
        faceEmbedder.close()
    }
}
