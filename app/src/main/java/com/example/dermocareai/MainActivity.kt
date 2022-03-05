package com.example.dermocareai

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.provider.MediaStore.Images.Media.getBitmap
import android.util.Log
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.example.dermocareai.ml.TfLiteModel
import kotlinx.android.synthetic.main.activity_main.*
import kotlinx.android.synthetic.main.ben_mal_layout.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer

class MainActivity : AppCompatActivity() {

    lateinit var bitmap: Bitmap

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        uploadImageButton.setOnClickListener {

            Intent(Intent.ACTION_GET_CONTENT).also {
                it.type = "image/*"
                startActivityForResult(it,100)
            }
        }

        benmalPredButton.setOnClickListener {
            if(imageView.drawable == null){
                Log.d("DIALOG","dialog box needed")
            }
            val fileName = "label.txt"
            val inputString = application.assets.open(fileName).bufferedReader().use{it.readText()}
            var townList = inputString.split("\n")

            //Log.d("checking",townList.size.toString())

            var resized : Bitmap = Bitmap.createScaledBitmap(bitmap,180,180,true)

            val model = TfLiteModel.newInstance(this)

            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 180, 180, 3), DataType.FLOAT32)

            var byteBuffer : ByteBuffer = ByteBuffer.allocateDirect(4*180*180*3)

//            Log.d("sizes_of_buffers","Byte Buffer"+byteBuffer.toString())
//            Log.d("sizes_of_buffers","input feature"+inputFeature0.buffer.toString())
            inputFeature0.loadBuffer(byteBuffer)

            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer
            var max = getMax(outputFeature0.floatArray)

            //Log.d("irriated",outputFeature0.floatArray[1].toString())

            var answer : String = townList[max]
            model.close()
            Intent(this,BenMalActivity::class.java).also {
                it.putExtra("EXTRA_ANSWER",answer)
                startActivity(it)
            }
        }
    }
    
    fun getMax(arr : FloatArray) : Int{
        if(arr[0]<arr[1])
            return 1
        return 0
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if(resultCode == Activity.RESULT_OK && requestCode == 100){
            val uri = data?.data
            imageView.setImageURI(uri)

            bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver,uri)
        }
    }
}
