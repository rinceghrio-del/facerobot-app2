package com.example.facerobot

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

/**
 * Nag-iimbak ng mga custom na voice command (trigger phrase -> sasabihing reply, at
 * opsyonal na ESP32 action gaya ng "LEFT"/"SPIN") na itina-type mismo ng user sa loob
 * ng app - gamit ang SharedPreferences bilang simpleng JSON, kagaya ng ginawa natin
 * sa FaceStore.
 */
class CommandStore(context: Context) {

    companion object {
        private const val PREFS_NAME = "command_store"
        private const val KEY_COMMANDS = "custom_commands_json"
    }

    // action = "" kung walang ipapadalang utos sa ESP32, magsasalita lang.
    data class VoiceCommand(val trigger: String, val reply: String, val action: String = "")

    private val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    private val commands = mutableListOf<VoiceCommand>()

    init {
        load()
    }

    private fun load() {
        commands.clear()
        val json = prefs.getString(KEY_COMMANDS, null) ?: return
        try {
            val array = JSONArray(json)
            for (i in 0 until array.length()) {
                val obj = array.getJSONObject(i)
                commands.add(
                    VoiceCommand(
                        obj.getString("trigger"),
                        obj.getString("reply"),
                        obj.optString("action", "") // "" kung wala pa dating action noon (lumang data)
                    )
                )
            }
        } catch (e: Exception) {
            // Kung sira yung saved JSON sa kadahilanang ano man, mag-start na lang tayo ulit
            // sa blangkong listahan imbes na mag-crash.
            commands.clear()
        }
    }

    private fun persist() {
        val array = JSONArray()
        for (cmd in commands) {
            val obj = JSONObject()
            obj.put("trigger", cmd.trigger)
            obj.put("reply", cmd.reply)
            obj.put("action", cmd.action)
            array.put(obj)
        }
        prefs.edit().putString(KEY_COMMANDS, array.toString()).apply()
    }

    /** Idinadagdag o pinapalitan (kung existing na ang trigger phrase) ang isang command. */
    fun add(trigger: String, reply: String, action: String = "") {
        val cleanTrigger = trigger.trim().lowercase()
        commands.removeAll { it.trigger == cleanTrigger }
        commands.add(VoiceCommand(cleanTrigger, reply.trim(), action.trim().uppercase()))
        persist()
    }

    fun remove(trigger: String) {
        commands.removeAll { it.trigger == trigger }
        persist()
    }

    fun all(): List<VoiceCommand> = commands.toList()

    /** Hinahanap ang unang command na "nakapaloob" sa sinabi ng user. Null kung wala. */
    fun findMatch(spokenText: String): VoiceCommand? {
        return commands.firstOrNull { spokenText.contains(it.trigger) }
    }
}
