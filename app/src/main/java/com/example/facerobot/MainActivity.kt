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
import android.os.Handler
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.text.InputType
import android.util.Log
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

    // ---- Palette (ginamit sa buong UI para magkatugma lahat) ----
    private val accentColor = 0xFF00E5C7.toInt()
    private val accentPressed = 0xFF00A896.toInt()
    private val darkChip = 0xFF1E1E2E.toInt()
    private val darkChipPressed = 0xFF2A2A3E.toInt()
    private val darkBg = 0xFF121212.toInt()
    private val dangerColor = 0xFFFF5C5C.toInt()
    private val dangerPressed = 0xFFCC4747.toInt()

    // ---------- Mic watchdog: para hindi na "mag-freeze" ang pakikinig ----------
    // Dating problema: kapag hindi na-fire ang callback ng SpeechRecognizer o ng TextToSpeech
    // (nangyayari minsan sa ibang device/OEM), permanenteng naka-true ang isListening/isSpeaking
    // kaya hindi na muling nakikinig ang mic. Ang mga watchdog runnable sa ibaba ay
    // pilit nagre-reset pagkalampas ng ilang segundo kung hindi natapos ang aksyon.
    private val mainHandler = Handler(Looper.getMainLooper())
    private var restartScheduled = false
    private var listenWatchdog: Runnable? = null
    private var speakWatchdog: Runnable? = null
    private val LISTEN_WATCHDOG_MS = 9000L
    private val SPEAK_WATCHDOG_MS = 7000L
    private var consecutiveClientErrors = 0

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
                        runOnUi { finishSpeaking() }
                    }
                    @Deprecated("Deprecated in Java")
                    override fun onError(utteranceId: String?) {
                        runOnUi { finishSpeaking() }
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
            statusText.text = "Naghahanap ng tao... (hinihintay permissions)"
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

        statusText = TextView(this).apply {
            setTextColor(0xFFFFFFFF.toInt())
            textSize = 13.5f
            letterSpacing = 0.015f
            setPadding(44, 24, 44, 24)
            gravity = Gravity.CENTER
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            elevation = 6f
            background = GradientDrawable().apply {
                setColor(0xF0121212.toInt())
                cornerRadius = 100f
                setStroke(2, 0x33FFFFFF)
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

    /** Iisang dark rounded "card" background para tugma lahat ng dialog sa itim na tema ng app. */
    private fun dialogCardBackground(radius: Float = 28f): Drawable = GradientDrawable().apply {
        setColor(darkBg)
        cornerRadius = radius
        setStroke(2, 0x22FFFFFF)
    }

    /** Themed EditText na dark at rounded, tugma sa itsura ng app (dati plain default EditText lang). */
    private fun themedInput(hintText: String): EditText = EditText(this).apply {
        hint = hintText
        inputType = InputType.TYPE_CLASS_TEXT
        setTextColor(0xFFFFFFFF.toInt())
        setHintTextColor(0xFF8A8A9A.toInt())
        textSize = 14f
        setPadding(32, 26, 32, 26)
        background = GradientDrawable().apply {
            setColor(darkChip)
            cornerRadius = 18f
            setStroke(2, 0x33FFFFFF)
        }
    }

    /** Ginagawang dark-themed ang alert dialog window mismo (hindi lang ang laman) para consistent. */
    private fun styleDialogWindow(dialog: android.app.AlertDialog) {
        dialog.window?.setBackgroundDrawable(GradientDrawable().apply {
            setColor(Color.TRANSPARENT)
        })
    }

    private fun showMainMenuDialog() {
        val disabledChip = 0xFF3A3A3A.toInt()

        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(40, 40, 40, 40)
            background = dialogCardBackground()
        }

        val title = TextView(this).apply {
            text = "Menu"
            setTextColor(0xFFFFFFFF.toInt())
            textSize = 16f
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            setPadding(8, 0, 8, 28)
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

        container.addView(title)
        container.addView(
            enrollOption,
            LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
        )
        container.addView(spacer)
        container.addView(
            commandsOption,
            LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
        )

        val dialog = android.app.AlertDialog.Builder(this)
            .setView(container)
            .setNegativeButton("Isara", null)
            .create()
        dialog.show()
        styleDialogWindow(dialog)
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

        // Kukunin lang ang LEFT/RIGHT o STOP (Paggitna)
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
        // Kung meron nang existing instance, wag nang gumawa ulit - ang paulit-ulit na
        // paggawa ng bagong SpeechRecognizer ang isa sa sanhi ng "pagka-freeze"/ERROR_CLIENT.
        if (speechRecognizer != null) {
            startListening()
            return
        }
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this).apply {
            setRecognitionListener(object : RecognitionListener {
                override fun onReadyForSpeech(params: Bundle?) {}
                override fun onBeginningOfSpeech() {}
                override fun onRmsChanged(rmsdB: Float) {}
                override fun onBufferReceived(buffer: ByteArray?) {}
                override fun onEndOfSpeech() {}
                override fun onError(error: Int) {
                    // Improved onError handling to aggressively recover from ERROR_CLIENT (11)
                    clearListenWatchdog()
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

                    // Log for easier debugging with adb logcat
                    Log.w("MainActivity", "SpeechRecognizer.onError: $error ($errorName)")

                    // If language not supported, fall back once
                    if (error == SpeechRecognizer.ERROR_LANGUAGE_NOT_SUPPORTED ||
                        error == SpeechRecognizer.ERROR_LANGUAGE_UNAVAILABLE
                    ) {
                        usingFallbackLocale = true
                    }

                    // For CLIENT / RECOGNIZER_BUSY errors: destroy and recreate immediately to avoid stale state
                    val needsRecreate = error == SpeechRecognizer.ERROR_CLIENT ||
                        error == SpeechRecognizer.ERROR_RECOGNIZER_BUSY

                    if (needsRecreate) {
                        consecutiveClientErrors = consecutiveClientErrors + 1

                        try {
                            // Best-effort to stop and destroy the possibly-broken recognizer instance
                            this@MainActivity.speechRecognizer?.cancel()
                        } catch (e: Exception) {
                            Log.w("MainActivity", "error while canceling recognizer", e)
                        }
                        try {
                            this@MainActivity.speechRecognizer?.destroy()
                        } catch (e: Exception) {
                            Log.w("MainActivity", "error while destroying recognizer", e)
                        }
                        this@MainActivity.speechRecognizer = null

                        statusText.text = "[MIC] Error: $errorName (recovering)"

                        // Schedule recreate; require 2 consecutive client errors before force recreate to avoid flapping
                        scheduleRestartListening(forceRecreate = true)
                        return
                    }

                    // Non-client errors: reset consecutive client error counter and restart listening (no recreate)
                    consecutiveClientErrors = 0
                    statusText.text = "[MIC] Error: $errorName"
                    scheduleRestartListening(forceRecreate = false)
                }
                override fun onResults(results: Bundle?) {
                    clearListenWatchdog()
                    isListening = false
                    consecutiveClientErrors = 0
                    val candidates = results
                        ?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                        ?.map { it.lowercase(Locale.getDefault()) }
                        .orEmpty()
                    statusText.text = if (candidates.isNotEmpty()) {
                        "[MIC] Narinig: ${candidates.joinToString(" / ")}"
                    } else {
                        "[MIC] Walang na-detect na salita"
                    }
                    // Sinusubukan lahat ng alternative na resulta, hindi lang yung pinaka-una,
                    // dahil minsan nasa 2nd o 3rd guess pa lang yung tamang tugma sa command.
                    if (candidates.isNotEmpty()) handleVoiceCommand(candidates)
                    scheduleRestartListening()
                }
                override fun onPartialResults(partialResults: Bundle?) {}
                override fun onEvent(eventType: Int, params: Bundle?) {}
            })
        }
        startListening()
    }

    // Dapat tugma ito sa locale na ginagamit ng TTS (Locale("fil", "PH")) - kung hindi,
    // maling language model ang gagamitin sa pakikinig kahit Filipino ang sinasabi ng user.
    private val recognitionLocale = Locale("fil", "PH")
    // Kapag na-detect na hindi supported ang Filipino sa device, gagamitin na lang
    // yung default locale ng phone (karaniwang mas malawak ang language support nito).
    private var usingFallbackLocale = false

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
            // BCP-47 tag (may dash, e.g. "fil-PH") ang inaasahan dito, hindi yung underscore
            // na output ng Locale.toString() - kaya toLanguageTag() ang ginagamit.
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, languageTag)
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, false)
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 3)
        }
        try {
            recognizer.startListening(intent)
            isListening = true
            armListenWatchdog()
        } catch (e: Exception) {
            isListening = false
            // Kapag nag-throw agad ang startListening, sirang state na ang recognizer -
            // ligtas nang gumawa ng bago sa susunod na restart imbes na paulit-ulit na sumigaw.
            scheduleRestartListening(forceRecreate = true)
        }
    }

    /** Watchdog: kung walang dumating na callback (onResults/onError) sa loob ng ilang segundo,
     * ituturing na natigil/na-freeze ang recognizer at pilit ire-restart. Dito galing ang fix
     * sa dating "minsan biglang hindi na nakikinig ang mic hanggang i-restart ang app". */
    private fun armListenWatchdog() {
        clearListenWatchdog()
        val watchdog = Runnable {
            if (isListening) {
                isListening = false
                statusText.text = "[MIC] Natigil, nagre-restart..."
                scheduleRestartListening(forceRecreate = true)
            }
        }
        listenWatchdog = watchdog
        mainHandler.postDelayed(watchdog, LISTEN_WATCHDOG_MS)
    }

    private fun clearListenWatchdog() {
        listenWatchdog?.let { mainHandler.removeCallbacks(it) }
        listenWatchdog = null
    }

    private fun armSpeakWatchdog() {
        clearSpeakWatchdog()
        val watchdog = Runnable {
            if (isSpeaking) {
                finishSpeaking()
            }
        }
        speakWatchdog = watchdog
        mainHandler.postDelayed(watchdog, SPEAK_WATCHDOG_MS)
    }

    private fun clearSpeakWatchdog() {
        speakWatchdog?.let { mainHandler.removeCallbacks(it) }
        speakWatchdog = null
    }

    private fun finishSpeaking() {
        clearSpeakWatchdog()
        isSpeaking = false
        scheduleRestartListening()
    }

    private fun scheduleRestartListening(forceRecreate: Boolean = false) {
        // Iisang naka-pending na restart lang sa isang pagkakataon - dati posibleng
        // magtambak ang mga restart (mula sa onError, onResults, atbp.) na siyang
        // nagdudulot ng sunod-sunod na paggawa ng recognizer at ERROR_CLIENT loop.
        if (restartScheduled) return
        restartScheduled = true
        mainHandler.postDelayed({
            restartScheduled = false
            if (forceRecreate) {
                try {
                    speechRecognizer?.destroy()
                } catch (e: Exception) {
                    // wala lang, tuloy pa rin tayo sa paggawa ng bago
                }
                speechRecognizer = null
            }
            setupSpeechRecognizer()
        }, 500)
    }

    private fun handleVoiceCommand(candidates: List<String>) {
        // Sinusubukan ang bawat alternative na resulta ng recognizer hanggang may tumama.
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
        clearListenWatchdog()
        speechRecognizer?.stopListening()
        isListening = false
        armSpeakWatchdog()
        tts?.speak(phrase, TextToSpeech.QUEUE_FLUSH, null, "utt_${System.currentTimeMillis()}")
    }

    private fun handleNoFace() {
        // Kapag walang mukha, hihinto lang at mag-aabang hanggang bumalik sa eyes mode
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

    /**
     * Tanging Panggitna/Centering logic na lang:
     * Sinusuri kung nasa Kaliwa, Kanan, o Gitna (STOP) ang mukha.
     */
    private fun computeCommand(faceCenterX: Int, frameWidth: Int): String {
        val screenCenterX = frameWidth / 2

        // Pinalapad ang deadzone (ginawang frameWidth / 3.5)
        // Mas malapad na gitnang espasyo para may allowance bago mag-STOP
        val centerDeadzoneWidth = (frameWidth / 3.5 / 2).toInt()

        val leftBoundary = screenCenterX - centerDeadzoneWidth
        val rightBoundary = screenCenterX + centerDeadzoneWidth

        return when {
            // Mirrored ang front camera input:
            // Kapag ang mukha ay nasa kaliwa sa pixel coordinates (faceCenterX < leftBoundary),
            // kailangang pumaling ng robot sa KANAN (RIGHT) para pumunta sa gitna ang mukha.
            faceCenterX < leftBoundary -> "LEFT"
            faceCenterX > rightBoundary -> "RIGHT"
            else -> "STOP" // Kapag pasok na sa deadzone, hihinto agad!
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
                // Connection fail error handling
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

        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(40, 40, 40, 40)
            background = dialogCardBackground()
        }
        val title = TextView(this).apply {
            text = "Mag-enroll ng mukha"
            setTextColor(0xFFFFFFFF.toInt())
            textSize = 16f
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            setPadding(8, 0, 8, 24)
        }
        val input = themedInput("Pangalan (hal. Rusty)")
        container.addView(title)
        container.addView(input)

        val dialog = android.app.AlertDialog.Builder(this)
            .setView(container)
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
            .create()
        dialog.show()
        styleDialogWindow(dialog)
    }

    private fun showManageCommandsDialog() {
        val outer = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(40, 40, 40, 32)
            background = dialogCardBackground()
        }

        val title = TextView(this).apply {
            text = "Mga Voice Command"
            setTextColor(0xFFFFFFFF.toInt())
            textSize = 16f
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            setPadding(8, 0, 8, 24)
        }
        outer.addView(title)

        val existing = commandStore.all()
        if (existing.isEmpty()) {
            outer.addView(TextView(this).apply {
                text = "Wala pang custom na command."
                setTextColor(0xFFAAAAAA.toInt())
                textSize = 13f
                setPadding(8, 0, 0, 24)
            })
        } else {
            for (cmd in existing) {
                val row = LinearLayout(this).apply {
                    orientation = LinearLayout.HORIZONTAL
                    gravity = Gravity.CENTER_VERTICAL
                    setPadding(24, 20, 20, 20)
                    background = GradientDrawable().apply {
                        setColor(darkChip)
                        cornerRadius = 16f
                    }
                    layoutParams = LinearLayout.LayoutParams(
                        LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
                    ).apply { bottomMargin = 16 }
                }
                row.addView(TextView(this@MainActivity).apply {
                    val actionPart = if (cmd.action.isNotBlank()) "  •  ESP32: ${cmd.action}" else ""
                    text = "\"${cmd.trigger}\" → \"${cmd.reply}\"$actionPart"
                    setTextColor(0xFFFFFFFF.toInt())
                    textSize = 13f
                    layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                })
                row.addView(Button(this@MainActivity).apply {
                    text = "Tanggalin"
                    textSize = 10f
                    setTextColor(0xFFFFFFFF.toInt())
                    isAllCaps = false
                    background = makeRippleRoundedDrawable(dangerColor, dangerPressed, 40f)
                    setOnClickListener {
                        commandStore.remove(cmd.trigger)
                        showManageCommandsDialog()
                    }
                })
                outer.addView(row)
            }
