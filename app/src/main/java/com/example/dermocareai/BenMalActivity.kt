package com.example.dermocareai

import android.graphics.Bitmap
import android.os.Bundle
import android.os.PersistableBundle
import androidx.appcompat.app.AppCompatActivity
import com.example.dermocareai.ml.TfLiteModel
import kotlinx.android.synthetic.main.activity_main.*
import kotlinx.android.synthetic.main.ben_mal_layout.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer

class BenMalActivity : AppCompatActivity() {


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.ben_mal_layout)

        val ans = intent.getStringExtra("EXTRA_ANSWER")
        benMalPredictTextView.text = ans

        homeButton1.setOnClickListener {
            finish()
        }
    }
}