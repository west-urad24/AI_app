import tkinter as tk
import tkinter.filedialog as fd
import PIL.Image
import PIL.ImageTk
import sklearn.datasets
import sklearn.svm
import numpy as np

# digits.images
# [[ 0.  0.  1. 14. 11.  0.  0.  0.]
#  [ 0.  0.  9. 15.  2.  0.  4.  0.]
#  [ 0.  2. 16.  6.  0.  7. 16.  2.]
#  [ 0.  8. 16.  6.  6. 16. 12.  0.]
#  [ 0.  5. 16. 16. 16. 15.  3.  0.]
#  [ 0.  0.  1.  4. 16.  8.  0.  0.]
#  [ 0.  0.  0.  9. 16.  1.  0.  0.]
#  [ 0.  0.  0. 15. 16.  0.  0.  0.]]
#digits.target
# 4

def imageToData(filename):#画像を数値に変換
    grayImage = PIL.Image.open(filename).convert("L")#白黒にする
    grayImage = grayImage.resize((8, 8), PIL.Image.ANTIALIAS)

    dispImage = PIL.ImageTk.PhotoImage(grayImage.resize((300,300),resample = 0))
    imageLabel.configure(image = dispImage)
    imageLabel.image = dispImage

    numImage = np.asarray(grayImage,dtype = float)
    numImage = 16 - np.floor(17*numImage/256)
    numImage = numImage.flatten()
    return numImage

def predictDigits(data):
    digits = sklearn.datasets.load_digits()
    clf = sklearn.svm.SVC(gamma = 0.001)#ハイパーパラメータ
    clf.fit(digits.data,digits.target)#学習

    n = clf.predict([data])#予測
    textLabel.configure(text = '予測結果は'+str(n)+'です。')

def openfile():
    fpath = fd.askopenfilename()#手書きの画像を読み込む
    if fpath:
        data = imageToData(fpath)
        predictDigits(data)#predictDigits関数に移動

root = tk.Tk()
root.geometry("400x400")

btn = tk.Button(root,text = "ファイルを開く",command = openfile)#ボタンとファイルの関数を繋げる
imageLabel = tk.Label()
btn.pack()
imageLabel.pack()

textLabel = tk.Label(text = '数字を認識します。「ファイルを開く」をクリックして下さい。')
textLabel.pack()

root.title("Easy_AI")
tk.mainloop()
