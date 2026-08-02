package com.example.facerobot

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Rect
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
    private lateinit var commandsButton: Button

    private lateinit var cameraExecutor: ExecutorService
    private val httpClient = OkHttpClient()

    private lateinit var yoloDetector: YoloPersonDetector
    private lateinit var faceEmbedder: FaceEmbedder
    private lateinit var faceStore: FaceStore
    private lateinit var commandStore: CommandStore

    private var appState = AppState.EYES

    // PALITAN ITO ng IP address ng ESP32 mo (makikita sa Serial Monitor pag nag-boot)
    private val esp32BaseUrl = "http://192.168.4.1"

    // Throttle para sa command papunta sa ESP32
    private var lastSendTime = 0L
    private val sendIntervalMs = 300L

    // Throttle para sa YOLO (mabigat siya kaysa ML Kit, kaya bihira lang patakbuhin)
    private var lastYoloCheckTime = 0L
    private val yoloIntervalMs = 400L

    // Ilang sunod-sunod na positibong detection na kailangan bago talaga lumipat sa
    // camera - iniiwasan nito yung false-positive na isang beses lang na "nakakita"
    // (ingay/shadow/blur) na nagbubukas ng camera kahit walang totoong tao.
    private var consecutivePersonDetections = 0
    private val requiredConsecutiveDetections = 3

    // Throttle para sa face recognition (embedding + compare)
    private var lastRecognitionTime = 0L
    private val recognitionIntervalMs = 600L

    // Kung gaano katagal walang nakikitang tao bago bumalik sa RoboEyes
    private var lastPersonSeenTime = 0L
    private val personTimeoutMs = 4000L

    // Huling nakuhang embedding ng "hindi kilalang" mukha - gagamitin ng enroll button
    private var lastUnknownFaceEmbedding: FloatArray? = null

    // Text-to-speech para sa pag-greet
    private var tts: TextToSpeech? = null
    private var ttsReady = false
    private var lastGreetedName: String? = null
    private var lastGreetedTime = 0L
    private val greetingCooldownMs = 60_000L // ulitin lang ang greeting kada 1 minuto per tao
    private var lastUnknownGreetTime = 0L

    // Voice command (hal. "Robot, sino ako?")
    private var speechRecognizer: SpeechRecognizer? = null
    private var isListening = false
    private var isSpeaking = false
    // Pangalan ng taong kasalukuyang nakikilala ng camera (null kung wala/hindi kilala)
    private var currentRecognizedName: String? = null

    private val faceDetectorOptions = FaceDetectorOptions.Builder()
        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
        .build()
    private val faceDetector = FaceDetection.getClient(faceDetectorOptions)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Hindi mamamatay ang screen habang bukas ang app
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

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
                // Subukan muna ang Filipino - kung wala sa device, English na lang
                val result = engine.setLanguage(Locale("fil", "PH"))
                if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    engine.setLanguage(Locale.US)
                }
                // Ipahinto ang pakikinig habang nagsasalita ang robot (iwas maka-dinig
                // ng sarili niyang boses), tapos ipagpatuloy pagkatapos.
                engine.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                    override fun onStart(utteranceId: String?) {}
                    override fun onDone(utteranceId: String?) {
                        isSpeaking = false
                        runOnUi { scheduleRestartListening() }
                    }
                    @Deprecated("Deprecated in Java")
                    override fun onError(utteranceId: String?) {
                        isSpeaking = false
                        runOnUi { scheduleRestartListening() }
                    }
                })
                ttsReady = true
            }
        }

        val missingPermissions = mutableListOf<String>()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            missingPermissions.add(Manifest.permission.CAMERA)
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
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

        commandsButton = Button(this).apply {
            text = "🎤 Mga Utos"
            textSize = 12f
            setOnClickListener { showManageCommandsDialog() }
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
        rootLayout.addView(
            commandsButton,
            FrameLayout.LayoutParams(FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT)
                .apply { gravity = Gravity.BOTTOM or Gravity.END; bottomMargin = 48; rightMargin = 24 }
        )

        setContentView(rootLayout)
    }

    private fun showEyesUi() {
        appState = AppState.EYES
        roboEyesView.visibility = View.VISIBLE
        // NOTE: INVISIBLE (hindi GONE) dapat dito - kapag GONE ang isang View, hindi ito
        // nire-layout/binibigyan ng laki, kaya walang wastong "surface" na maibibigay sa
        // CameraX Preview use case. Kapag ganito, humihintay nang walang hanggan ang buong
        // camera capture session (kasama ang ImageAnalysis) para sa surface na 'yon - kaya
        // hindi umaandar ang camera kahit "successful" ang bind (walang exception). INVISIBLE
        // pa rin ang laki/layout, walang lang draw sa screen - RoboEyes pa rin ang makikita.
        previewView.visibility = View.INVISIBLE
        enrollButton.visibility = View.GONE
        roboEyesView.setMood(RoboEyesView.Mood.SEARCHING)
        statusText.text = if (yoloDetector.isReady) {
            "Naghahanap ng tao..."
        } else {
            "Naghahanap ng tao... (kulang: assets/yolo_person.tflite)"
        }
        // Bawat pagbalik sa EYES (tao'y umalis) - pwede na ulit mag-greet sa susunod na makita
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
            // Kung tinanggihan ang mic permission, tuloy pa rin ang app - wala lang voice command.
        }
    }

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

            if (detections.isNotEmpty()) {
                consecutivePersonDetections++
            } else {
                consecutivePersonDetections = 0
            }

            if (consecutivePersonDetections >= requiredConsecutiveDetections) {
                runOnUi { roboEyesView.setMood(RoboEyesView.Mood.ALERT) }
                // Konting delay para makita muna ang "alert" na expression bago lumipat
                rootLayout.postDelayed({
                    // Recheck: baka nag-reset na (person left) bago pa dumating ang delay
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
                                currentRecognizedName = match.name
                                greetIfNeeded(match.name)
                            } else {
                                statusText.text = if (faceStore.isEmpty()) {
                                    "May tao pero wala pang naka-enroll na mukha"
                                } else {
                                    "Hindi kilala"
                                }
                                lastUnknownFaceEmbedding = embedding
                                enrollButton.visibility = View.VISIBLE
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

    /** Mag-greet lang minsan kada tao, tapos maghintay ng cooldown bago ulitin. */
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

    /** Parehong cooldown-logic pero para sa mga hindi pa naka-enroll na mukha. */
    private fun greetUnknownIfNeeded() {
        val now = System.currentTimeMillis()
        if (now - lastUnknownGreetTime < greetingCooldownMs) return
        lastUnknownGreetTime = now

        if (!ttsReady) return
        speak(unknownGreetings.random())
    }

    // ---------- Voice command ----------

    private fun setupSpeechRecognizer() {
        if (!SpeechRecognizer.isRecognitionAvailable(this)) {
            // Walang speech recognition sa device na 'to - hindi mag-crash, wala lang
            // voice command feature.
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
                    isListening = false
                    scheduleRestartListening()
                }
                override fun onResults(results: Bundle?) {
                    isListening = false
                    val text = results
                        ?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                        ?.firstOrNull()
                        ?.lowercase(Locale.getDefault())
                    if (text != null) handleVoiceCommand(text)
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
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, false)
        }
        isListening = true
        try {
            recognizer.startListening(intent)
        } catch (e: Exception) {
            isListening = false
        }
    }

    private fun scheduleRestartListening() {
        rootLayout.postDelayed({ startListening() }, 500)
    }

    private fun handleVoiceCommand(text: String) {
        // Unahin munang suriin ang mga custom na command na idinagdag mismo sa app
        val custom = commandStore.findMatch(text)
        if (custom != null) {
            speak(custom.reply)
            if (custom.action.isNotBlank()) {
                sendCommandToEsp32(custom.action)
            }
            return
        }

        when {
            text.contains("sino ako") || text.contains("sino po ako") || text.contains("sino ba ako") -> {
                val name = currentRecognizedName
                val reply = when {
                    name != null -> "Ikaw ay si $name!"
                    appState == AppState.CAMERA -> "Hindi pa kita kilala. Pwede mo akong i-enroll."
                    else -> "Wala akong nakikitang tao ngayon."
                }
                speak(reply)
            }
            text.contains("sino ka") -> {
                speak("ako ay si rustech")
            }
        }
    }

    /** Sentralisadong paraan para magsalita - ipinapahinto muna ang mic bago mag-TTS. */
    private fun speak(phrase: String) {
        if (!ttsReady) return
        isSpeaking = true
        speechRecognizer?.stopListening()
        isListening = false
        tts?.speak(phrase, TextToSpeech.QUEUE_FLUSH, null, "utt_${System.currentTimeMillis()}")
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

    /** Dialog kung saan pwedeng magdagdag/magtanggal ng custom na voice command sa app mismo. */
    private fun showManageCommandsDialog() {
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 24, 48, 24)
        }

        val existing = commandStore.all()
        if (existing.isEmpty()) {
            container.addView(TextView(this).apply {
                text = "Wala pang custom na command."
                setPadding(0, 0, 0, 24)
            })
        } else {
            for (cmd in existing) {
                val row = LinearLayout(this).apply {
                    orientation = LinearLayout.HORIZONTAL
                    gravity = Gravity.CENTER_VERTICAL
                }
                row.addView(TextView(this@MainActivity).apply {
                    val actionPart = if (cmd.action.isNotBlank()) " [ESP32: ${cmd.action}]" else ""
                    text = "\"${cmd.trigger}\" -> \"${cmd.reply}\"$actionPart"
                    textSize = 13f
                    layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                })
                row.addView(Button(this@MainActivity).apply {
                    text = "Tanggalin"
                    textSize = 10f
                    setOnClickListener {
                        commandStore.remove(cmd.trigger)
                        showManageCommandsDialog() // i-refresh ang dialog
                    }
                })
                container.addView(row)
            }
        }

        container.addView(View(this).apply {
            setBackgroundColor(0xFFCCCCCC.toInt())
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 2)
                .apply { topMargin = 32; bottomMargin = 32 }
        })

        container.addView(TextView(this).apply { text = "Magdagdag ng bagong command:" })

        val triggerInput = EditText(this).apply {
            hint = "Sasabihin (hal. anong oras na)"
            inputType = InputType.TYPE_CLASS_TEXT
        }
        val replyInput = EditText(this).apply {
            hint = "Isasagot ng robot"
            inputType = InputType.TYPE_CLASS_TEXT
        }
        val actionInput = EditText(this).apply {
            hint = "ESP32 action (opsyonal - hal. LEFT, RIGHT, SPIN, STOP - iwanan blangko kung wala)"
            inputType = InputType.TYPE_CLASS_TEXT
        }
        container.addView(triggerInput)
        container.addView(replyInput)
        container.addView(actionInput)

        val scrollView = ScrollView(this).apply { addView(container) }

        android.app.AlertDialog.Builder(this)
            .setTitle("Mga Voice Command")
            .setView(scrollView)
            .setPositiveButton("Idagdag") { _, _ ->
                val trigger = triggerInput.text.toString().trim()
                val reply = replyInput.text.toString().trim()
                val action = actionInput.text.toString().trim()
                if (trigger.isNotEmpty() && reply.isNotEmpty()) {
                    commandStore.add(trigger, reply, action)
                    statusText.text = "Idinagdag na command: \"$trigger\""
                }
            }
            .setNegativeButton("Isara", null)
            .show()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        faceDetector.close()
        yoloDetector.close()
        faceEmbedder.close()
        tts?.stop()
        tts?.shutdown()
        speechRecognizer?.destroy()
    }
}
