package com.example.facerobot

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Color
import android.graphics.Rect
import android.graphics.Typeface
import android.graphics.drawable.Drawable
import android.graphics.drawable.GradientDrawable
import android.graphics.drawable.RippleDrawable
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.net.NetworkRequest
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.text.InputType
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ScrollView
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
import java.util.Locale
import java.util.concurrent.Executors

/**
 * FaceRobot MainActivity - Face Centering / Tracking Only Mode
 */
@androidx.camera.core.ExperimentalGetImage
class MainActivity : ComponentActivity() {

    private enum class AppState { EYES, CAMERA }

    private lateinit var rootLayout: FrameLayout
    private lateinit var roboEyesView: RoboEyesView
    private lateinit var previewView: PreviewView
    private lateinit var statusText: TextView
    private lateinit var menuButton: Button
    private var canEnroll = false

    private lateinit var cameraExecutor: ExecutorService
    private val httpClient = OkHttpClient()

    private lateinit var yoloDetector: YoloPersonDetector
    private lateinit var faceEmbedder: FaceEmbedder
    private lateinit var faceStore: FaceStore
    private lateinit var commandStore: CommandStore

    private var appState = AppState.EYES

    private val esp32BaseUrl = "http://192.168.1.184"

    private var lastSendTime = 0L
    private val sendIntervalMs = 300L

    private var lastYoloCheckTime = 0L
    private val yoloIntervalMs = 400L

    private var consecutivePersonDetections = 0
    private val requiredConsecutiveDetections = 3

    private var lastRecognitionTime = 0L
    private val recognitionIntervalMs = 600L

    private var lastPersonSeenTime = 0L
    private val personTimeoutMs = 4000L

    private var lastUnknownFaceEmbedding: FloatArray? = null

    private var tts: TextToSpeech? = null
    private var ttsReady = false
    private var lastGreetedName: String? = null
    private var lastGreetedTime = 0L
    private val greetingCooldownMs = 60_000L
    private var lastUnknownGreetTime = 0L

    private var speechRecognizer: SpeechRecognizer? = null
    private var isListening = false
    private var isSpeaking = false
    private var currentRecognizedName: String? = null

    private val recognitionLocale = Locale("fil", "PH")
    private var usingFallbackLocale = false

    private val faceDetectorOptions = FaceDetectorOptions.Builder()
        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
        .build()
    private val faceDetector = FaceDetection.getClient(faceDetectorOptions)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        forceWifiForEsp32()

        cameraExecutor = Executors.newSingleThreadExecutor()
        yoloDetector = YoloPersonDetector(this)
        faceEmbedder = FaceEmbedder(this)
        faceStore = FaceStore(this)
        commandStore = CommandStore(this)

        buildUi()
        showEyesUi()

        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val engine = tts ?: return@TextToSpeech
                val result = engine.setLanguage(Locale("fil", "PH"))
                if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    engine.setLanguage(Locale.US)
                }
                engine.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                    override fun onStart(utteranceId: String?) {}
                    override fun onDone(utteranceId: String?) {
                        isSpeaking = false
                        // Naglagay ng 400ms delay para maka-recover ang audio hardware
                        runOnUi {
                            rootLayout.postDelayed({ scheduleRestartListening() }, 400)
                        }
                    }
                    @Deprecated("Deprecated in Java")
                    override fun onError(utteranceId: String?) {
                        isSpeaking = false
                        runOnUi {
                            rootLayout.postDelayed({ scheduleRestartListening() }, 400)
                        }
                    }
                })
                ttsReady = true
            }
        }

        val missingPermissions = mutableListOf<String>()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            missingPermissions.add(Manifest.permission.CAMERA)
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            missingPermissions.add(Manifest.permission.RECORD_AUDIO)
        }

        if (missingPermissions.isEmpty()) {
            startCamera()
            setupSpeechRecognizer()
        } else {
            statusText.text = "Naghahanap ng tao... (hinihintay permissions...)"
            ActivityCompat.requestPermissions(this, missingPermissions.toTypedArray(), 100)
        }
    }

    private fun forceWifiForEsp32() {
        try {
            val connectivityManager = getSystemService(CONNECTIVITY_SERVICE) as ConnectivityManager
            val request = NetworkRequest.Builder()
                .addTransportType(NetworkCapabilities.TRANSPORT_WIFI)
                .build()

            connectivityManager.requestNetwork(request, object : ConnectivityManager.NetworkCallback() {
                override fun onAvailable(network: Network) {
                    connectivityManager.bindProcessToNetwork(network)
                }
            })
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    // ---------- UI setup ----------

    private fun buildUi() {
        rootLayout = FrameLayout(this)
        roboEyesView = RoboEyesView(this)
        previewView = PreviewView(this)

        val accentColor = 0xFF00E5C7.toInt()
        val darkChip = 0xFF1E1E2E.toInt()
        val darkChipPressed = 0xFF2A2A3E.toInt()

        statusText = TextView(this).apply {
            setTextColor(0xFFFFFFFF.toInt())
            textSize = 13f
            setPadding(40, 22, 40, 22)
            gravity = Gravity.CENTER
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            background = GradientDrawable().apply {
                setColor(0xE6121212.toInt())
                cornerRadius = 100f
                setStroke(2, 0x22FFFFFF)
            }
        }

        menuButton = Button(this).apply {
            text = "☰"
            textSize = 20f
            setTextColor(accentColor)
            setPadding(0, 0, 0, 0)
            stateListAnimator = null
            elevation = 10f
            background = makeRippleRoundedDrawable(darkChip, darkChipPressed, 200f)
            setOnClickListener { showMainMenuDialog() }
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
            FrameLayout.LayoutParams(FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT)
                .apply { gravity = Gravity.TOP or Gravity.CENTER_HORIZONTAL; topMargin = 40 }
        )
        rootLayout.addView(
            menuButton,
            FrameLayout.LayoutParams(150, 150)
                .apply { gravity = Gravity.BOTTOM or Gravity.END; bottomMargin = 32; rightMargin = 24 }
        )

        setContentView(rootLayout)
    }

    private fun makeRippleRoundedDrawable(baseColor: Int, pressedColor: Int, radius: Float): Drawable {
        val shape = GradientDrawable().apply {
            setColor(baseColor)
            cornerRadius = radius
        }
        val mask = GradientDrawable().apply {
            setColor(Color.WHITE)
            cornerRadius = radius
        }
        return RippleDrawable(ColorStateList.valueOf(0x40FFFFFF), shape, mask)
    }

    private fun showMainMenuDialog() {
        val accentColor = 0xFF00E5C7.toInt()
        val accentPressed = 0xFF00A896.toInt()
        val darkChip = 0xFF1E1E2E.toInt()
        val darkChipPressed = 0xFF2A2A3E.toInt()
        val disabledChip = 0xFF3A3A3A.toInt()

        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(40, 40, 40, 32)
            setBackgroundColor(0xFF121212.toInt())
        }

        val enrollOption = Button(this).apply {
            text = "✨  Mag-enroll ng bagong mukha"
            textSize = 14f
            isAllCaps = false
            gravity = Gravity.START or Gravity.CENTER_VERTICAL
            setPadding(40, 36, 40, 36)
            isEnabled = canEnroll
            if (canEnroll) {
                setTextColor(0xFF04342C.toInt())
                background = makeRippleRoundedDrawable(accentColor, accentPressed, 24f)
            } else {
                setTextColor(0xFF888888.toInt())
                background = GradientDrawable().apply { setColor(disabledChip); cornerRadius = 24f }
            }
            setOnClickListener { showEnrollDialog() }
        }

        val commandsOption = Button(this).apply {
            text = "🎤  Mga Utos"
            textSize = 14f
            isAllCaps = false
            setTextColor(0xFFFFFFFF.toInt())
            gravity = Gravity.START or Gravity.CENTER_VERTICAL
            setPadding(40, 36, 40, 36)
            background = makeRippleRoundedDrawable(darkChip, darkChipPressed, 24f)
            setOnClickListener { showManageCommandsDialog() }
        }

        val spacer = View(this).apply {
            layoutParams = LinearLayout.LayoutParams(0, 24)
        }

        container.addView(
            enrollOption,
            LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
        )
        container.addView(spacer)
        container.addView(
            commandsOption,
            LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
        )

        android.app.AlertDialog.Builder(this)
            .setView(container)
            .setNegativeButton("Isara", null)
            .show()
    }

    private fun showEyesUi() {
        appState = AppState.EYES
        roboEyesView.visibility = View.VISIBLE
        previewView.visibility = View.INVISIBLE
        canEnroll = false
        roboEyesView.setMood(RoboEyesView.Mood.SEARCHING)
        statusText.text = if (yoloDetector.isReady) {
            "Naghahanap ng tao..."
        } else {
            "Naghahanap ng tao... (kulang: assets/yolo_person.tflite)"
        }
        lastGreetedName = null
        lastUnknownGreetTime = 0L
        consecutivePersonDetections = 0
        currentRecognizedName = null
    }

    private fun showCameraUi() {
        appState = AppState.CAMERA
        roboEyesView.visibility = View.GONE
        previewView.visibility = View.VISIBLE
        lastPersonSeenTime = System.currentTimeMillis()
        statusText.text = "May tao! Sinusubukang kilalanin..."
    }

    private fun runOnUi(block: () -> Unit) = runOnUiThread(block)

    // ---------- Camera setup ----------

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
        if (requestCode == 100) {
            val grantedMap = permissions.zip(grantResults.toList()).toMap()

            if (grantedMap[Manifest.permission.CAMERA] == PackageManager.PERMISSION_GRANTED) {
                startCamera()
            } else if (permissions.contains(Manifest.permission.CAMERA)) {
                statusText.text = "Naghahanap ng tao... (TINANGGIHAN ang camera permission)"
            }

            if (grantedMap[Manifest.permission.RECORD_AUDIO] == PackageManager.PERMISSION_GRANTED) {
                setupSpeechRecognizer()
            }
        }
    }

    private fun processFrame(imageProxy: ImageProxy) {
        when (appState) {
            AppState.EYES -> processEyesFrame(imageProxy)
            AppState.CAMERA -> processCameraFrame(imageProxy)
        }
    }

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

            if (detections.isNotEmpty()) {
                consecutivePersonDetections++
            } else {
                consecutivePersonDetections = 0
            }

            if (consecutivePersonDetections >= requiredConsecutiveDetections) {
                runOnUi { roboEyesView.setMood(RoboEyesView.Mood.ALERT) }
                rootLayout.postDelayed({
                    if (appState == AppState.EYES && consecutivePersonDetections >= requiredConsecutiveDetections) {
                        showCameraUi()
                    }
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

        val command = computeCommand(box.centerX(), frameWidth)
        sendCommandThrottled(command)

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
                                canEnroll = false
                                lastUnknownFaceEmbedding = null
                                currentRecognizedName = match.name
                                greetIfNeeded(match.name)
                            } else {
                                statusText.text = if (faceStore.isEmpty()) {
                                    "May tao pero wala pang naka-enroll na mukha"
                                } else {
                                    "Hindi kilala"
                                }
                                lastUnknownFaceEmbedding = embedding
                                canEnroll = true
                                currentRecognizedName = null
                                greetUnknownIfNeeded()
                            }
                        }
                    }
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    private val greetings = listOf(
        "Kumusta, %s!",
        "Hi %s, kumusta ka?",
        "Ay, si %s! Kumusta?",
        "Magandang araw, %s!"
    )

    private val unknownGreetings = listOf(
        "Kumusta, hindi pa kita kilala.",
        "Hi stranger! Hindi pa tayo nagkakakilala.",
        "Kumusta! Sino ka nga pala?",
        "Hello! Pwede mo ba akong pa-enroll?"
    )

    private fun greetIfNeeded(name: String) {
        val now = System.currentTimeMillis()
        val alreadyGreetedRecently = name == lastGreetedName && now - lastGreetedTime < greetingCooldownMs
        if (alreadyGreetedRecently) return

        lastGreetedName = name
        lastGreetedTime = now

        if (!ttsReady) return
        val phrase = greetings.random().format(name)
        speak(phrase)
    }

    private fun greetUnknownIfNeeded() {
        val now = System.currentTimeMillis()
        if (now - lastUnknownGreetTime < greetingCooldownMs) return
        lastUnknownGreetTime = now

        if (!ttsReady) return
        speak(unknownGreetings.random())
    }

    // ---------- Voice command ----------

    private fun setupSpeechRecognizer() {
        if (!SpeechRecognizer.isRecognitionAvailable(this)) return
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this).apply {
            setRecognitionListener(object : RecognitionListener {
                override fun onReadyForSpeech(params: Bundle?) {}
                override fun onBeginningOfSpeech() {}
                override fun onRmsChanged(rmsdB: Float) {}
                override fun onBufferReceived(buffer: ByteArray?) {}
                override fun onEndOfSpeech() {}
                override fun onError(error: Int) {
                    isListening = false
                    val errorName = when (error) {
                        SpeechRecognizer.ERROR_NO_MATCH -> "WALANG NARINIG NA SALITA"
                        SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "TAHIMIK LANG / WALANG NAGSALITA"
                        SpeechRecognizer.ERROR_LANGUAGE_NOT_SUPPORTED -> "HINDI SUPORTADO ANG FILIPINO SA PHONE MO"
                        SpeechRecognizer.ERROR_LANGUAGE_UNAVAILABLE -> "FILIPINO LANGUAGE PACK WALA / DI AVAILABLE"
                        SpeechRecognizer.ERROR_AUDIO -> "AUDIO ERROR (mic)"
                        SpeechRecognizer.ERROR_CLIENT -> "CLIENT ERROR"
                        SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "WALANG MIC PERMISSION"
                        SpeechRecognizer.ERROR_NETWORK -> "NETWORK ERROR"
                        SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> "NETWORK TIMEOUT"
                        SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "BUSY PA YUNG RECOGNIZER"
                        SpeechRecognizer.ERROR_SERVER -> "SERVER ERROR"
                        else -> "ERROR CODE $error"
                    }

                    if (error == SpeechRecognizer.ERROR_LANGUAGE_NOT_SUPPORTED ||
                        error == SpeechRecognizer.ERROR_LANGUAGE_UNAVAILABLE
                    ) {
                        usingFallbackLocale = true
                    }
                    statusText.text = "[MIC] Error: $errorName"

                    // Binigyan ngmas mahabang allowance kapag BUSY para hindi mag-looping error
                    val restartDelay = if (error == SpeechRecognizer.ERROR_RECOGNIZER_BUSY) 1200L else 600L
                    rootLayout.postDelayed({ scheduleRestartListening() }, restartDelay)
                }

                override fun onResults(results: Bundle?) {
                    isListening = false
                    val candidates = results
                        ?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                        ?.map { it.lowercase(Locale.getDefault()) }
                        .orEmpty()
                    statusText.text = if (candidates.isNotEmpty()) {
                        "[MIC] Narinig: ${candidates.joinToString(" / ")}"
                    } else {
                        "[MIC] Walang na-detect na salita"
                    }

                    if (candidates.isNotEmpty()) handleVoiceCommand(candidates)
                    scheduleRestartListening()
                }

                override fun onPartialResults(partialResults: Bundle?) {}
                override fun onEvent(eventType: Int, params: Bundle?) {}
            })
        }
        startListening()
    }

    private fun startListening() {
        if (isListening || isSpeaking) return
        val recognizer = speechRecognizer ?: return
        val languageTag = if (usingFallbackLocale) {
            Locale.getDefault().toLanguageTag()
        } else {
            recognitionLocale.toLanguageTag()
        }
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, languageTag)
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, false)
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 3)

            // Parameter para pahabain ang silence threshold bago mag-timeout
            putExtra("android.speech.extra.DICTATION_MODE", true)
            putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS, 1500L)
            putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_POSSIBLY_COMPLETE_SILENCE_LENGTH_MILLIS, 1500L)
        }
        isListening = true
        try {
            recognizer.startListening(intent)
        } catch (e: Exception) {
            isListening = false
        }
    }

    private fun scheduleRestartListening() {
        try {
            speechRecognizer?.destroy()
        } catch (e: Exception) {
            // Safe ignore
        }
        speechRecognizer = null
        setupSpeechRecognizer()
    }

    private fun handleVoiceCommand(candidates: List<String>) {
        for (text in candidates) {
            val custom = commandStore.findMatch(text)
            if (custom != null) {
                speak(custom.reply)
                if (custom.action.isNotBlank()) {
                    sendCommandToEsp32(custom.action)
                }
                return
            }

            when {
                // Motion Voice Commands
                text.contains("hinto") || text.contains("stop") || text.contains("tigil") -> {
                    speak("Hihinto na po!")
                    sendCommandToEsp32("STOP")
                    return
                }
                text.contains("kaliwa") || text.contains("left") -> {
                    speak("Lilikot sa kaliwa.")
                    sendCommandToEsp32("LEFT")
                    return
                }
                text.contains("kanan") || text.contains("right") -> {
                    speak("Lilikot sa kanan.")
                    sendCommandToEsp32("RIGHT")
                    return
                }

                // Info Voice Commands
                text.contains("sino ako") || text.contains("sino po ako") || text.contains("sino ba ako") -> {
                    val name = currentRecognizedName
                    val reply = when {
                        name != null -> "Ikaw ay si $name!"
                        appState == AppState.CAMERA -> "Hindi pa kita kilala. Pwede mo akong i-enroll."
                        else -> "Wala akong nakikitang tao ngayon."
                    }
                    speak(reply)
                    return
                }
                text.contains("sino ka") -> {
                    speak("ako ay si rustech")
                    return
                }
            }
        }
    }

    private fun speak(phrase: String) {
        if (!ttsReady) return
        isSpeaking = true
        speechRecognizer?.stopListening()
        isListening = false
        tts?.speak(phrase, TextToSpeech.QUEUE_FLUSH, null, "utt_${System.currentTimeMillis()}")
    }

    private fun handleNoFace() {
        sendCommandThrottled("STOP")
        val now = System.currentTimeMillis()
        if (now - lastPersonSeenTime > personTimeoutMs) {
            runOnUi { showEyesUi() }
        }
    }

    private fun adjustBoxForRotation(box: Rect, frameWidth: Int, frameHeight: Int, rotationDegrees: Int): Rect {
        return when (rotationDegrees) {
            90 -> Rect(frameHeight - box.bottom, box.left, frameHeight - box.top, box.right)
            180 -> Rect(frameWidth - box.right, frameHeight - box.bottom, frameWidth - box.left, frameHeight - box.top)
            270 -> Rect(box.top, frameWidth - box.right, box.bottom, frameWidth - box.left)
            else -> box
        }
    }

    private fun computeCommand(faceCenterX: Int, frameWidth: Int): String {
        val screenCenterX = frameWidth / 2
        val centerDeadzoneWidth = (frameWidth / 3.5 / 2).toInt()

        val leftBoundary = screenCenterX - centerDeadzoneWidth
        val rightBoundary = screenCenterX + centerDeadzoneWidth

        return when {
            faceCenterX < leftBoundary -> "LEFT"
            faceCenterX > rightBoundary -> "RIGHT"
            else -> "STOP"
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
                // Connection failure handling
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
                    canEnroll = false
                    lastUnknownFaceEmbedding = null
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun showManageCommandsDialog() {
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 32, 48, 32)
        }

        val title = TextView(this).apply {
            text = "Mga Custom Voice Command"
            textSize = 18f
            setTypeface(null, Typeface.BOLD)
            setPadding(0, 0, 0, 24)
        }
        container.addView(title)

        val scrollView = ScrollView(this)
        val listLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }

        fun refreshList() {
            listLayout.removeAllViews()
            val commands = commandStore.getAll()
            if (commands.isEmpty()) {
                val emptyTv = TextView(this).apply {
                    text = "Wala pang naisave na command."
                    setPadding(0, 16, 0, 16)
                }
                listLayout.addView(emptyTv)
            } else {
                for (cmd in commands) {
                    val row = LinearLayout(this).apply {
                        orientation = LinearLayout.HORIZONTAL
                        setPadding(0, 8, 0, 8)
                    }
                    val tv = TextView(this).apply {
                        text = "• \"${cmd.phrase}\" -> \"${cmd.reply}\" ${if (cmd.action.isNotBlank()) "[${cmd.action}]" else ""}"
                        layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                    }
                    val delBtn = Button(this).apply {
                        text = "✕"
                        setTextColor(Color.RED)
                        setOnClickListener {
                            commandStore.delete(cmd.phrase)
                            refreshList()
                        }
                    }
                    row.addView(tv)
                    row.addView(delBtn)
                    listLayout.addView(row)
                }
            }
        }

        refreshList()
        scrollView.addView(listLayout)
        container.addView(scrollView, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 400))

        val addBtn = Button(this).apply {
            text = "+ Magdagdag ng Command"
            setOnClickListener { showAddCommandDialog { refreshList() } }
        }
        container.addView(addBtn)

        android.app.AlertDialog.Builder(this)
            .setView(container)
            .setPositiveButton("OK", null)
            .show()
    }

    private fun showAddCommandDialog(onAdded: () -> Unit) {
        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 24, 48, 24)
        }

        val phraseInput = EditText(this).apply { hint = "Salita (e.g. kumusta)" }
        val replyInput = EditText(this).apply { hint = "Sagot ng robot (e.g. Mabuti naman)" }
        val actionInput = EditText(this).apply { hint = "ESP32 Action (e.g. LEFT, RIGHT, STOP o leave blank)" }

        layout.addView(phraseInput)
        layout.addView(replyInput)
        layout.addView(actionInput)

        android.app.AlertDialog.Builder(this)
            .setTitle("Bagong Command")
            .setView(layout)
            .setPositiveButton("Save") { _, _ ->
                val p = phraseInput.text.toString().trim()
                val r = replyInput.text.toString().trim()
                val a = actionInput.text.toString().trim()
                if (p.isNotEmpty() && r.isNotEmpty()) {
                    commandStore.add(p, r, a)
                    onAdded()
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        tts?.stop()
        tts?.shutdown()
        speechRecognizer?.destroy()
    }
}
