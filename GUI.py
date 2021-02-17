from tkinter import *
from PIL import ImageTk, Image
import imageio
from Prototype import *
import pyttsx3
import speech_recognition
from Execute import *

a= CONVAI()
b= Execute()

def stream():
    try:
        image = video.get_next_data()
        frame_image = Image.fromarray(image)
        frame_image = frame_image.resize((1000, 778), Image.ANTIALIAS)
        frame_image=ImageTk.PhotoImage(frame_image)
        l1.config(image=frame_image)
        l1.image = frame_image
        l1.after(delay, lambda: stream())
    except:
        video.close()
        return


    
window = Tk()
window.geometry("2000x1200")
window.title('Conversational AI')
window["bg"]="darkgreen"



# frame = Frame(window,height=480,width=600, bg="blue")
# frame.place(x=10, y=10)

f1=Frame()
l1 = Label(f1)
x = "D:\Downloads\AI_model.mp4" 
l1.place(x=0,y=0)
f1.place(x=10, y=10, height=778, width=1000)
video_name = x  #Image-path
video = imageio.get_reader(video_name)
delay = int(1000 / video.get_meta_data()['fps'])

stream()

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
# print(voices[1].id)
engine.setProperty('voice', voices[1].id)


def speak(audio):
    engine.say(audio)
    engine.runAndWait()
    
def train():
    a.no = int(num.get())
    a.init()


   

def send():
    # a.init()
    send="YOU=>"+e.get()
    print(send)
    ar = b.Test(e.get())
    print(ar)
    speak(ar)
    ar = str(ar)+"<="
    ar = ar.rjust(85)
    txt.insert(END, "\n"+send)
    txt.insert(END, "\n"+ar)
    e.delete(0,END)
       
    
txt = Text(window,bd=5, bg="lightblue")
txt.place(x=825,y=10,height=778, width=698)
e = Entry(window,width=86,bd=5)
e.place(x=830,y=757)
num = Entry(window,width=15,bd=5, bg="lightyellow")
num.place(x=1250,y=757)
send = Button(window,text ='Send',command=send,width=10,bg="lightgreen")
window.bind('<Return>', lambda event=None: send.invoke())
send.place(x=1440,y=757)
train = Button(window,text ='Train',command=train,width=10,bg="lightyellow")
train.place(x=1356,y=757)


window.mainloop()