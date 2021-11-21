import cv2
import numpy as np
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.logger import Logger
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy_garden.xcamera import XCamera
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivymd.app import MDApp as App
from kivy.graphics.texture import Texture
from kivy.uix.textinput import TextInput
from pat_class import Patient
from detection_cla import detectionsss

Logger.info(f"Versions: Numpy {np.__version__}")
Logger.info(f"Versions: Opencv {cv2.__version__}")

class CameraCV(XCamera):
    def on_tex(self, *l):
        self.image_bytes = self._camera.texture.pixels
        self.image_size = self._camera.texture.size

class CamApp(App):

    def welcome_callback(self, instance):
        self.root.current='main_screen'

    def capture_callback(self, instance):
        cv2.imwrite("image.jpeg", self.imggg)
        det = detectionsss()
        det.run()
        self.root.current = 'detection_screen'

    def welcome_screen(self,W_Screen):
        layout = FloatLayout(size=Window.size)
        button = Button(text="Hello, get ready to disgust yourself", size_hint=(0.4, 0.2),
                             pos_hint={'center_x': 0.5, 'center_y': 0.5})
        button.bind(on_press= self.welcome_callback)
        layout.add_widget(button)
        W_Screen.add_widget(layout)

    def main_screen(self, M_screen):
        self.size_ratio = None
        self.input_size = (320, 240)
        self.width = self.input_size[0]
        self.height = self.input_size[1]
        self.check_window_size()
        self.img1 = Image(pos_hint={'center_x': 0.5, 'center_y': 0.5})
        self.cap_button = Button(text="Capture picture", size_hint=(0.2, 0.4),
                                  pos_hint={'center_x': 0.75, 'center_y': 0.125})
        self.cap_button.bind(on_press=self.capture_callback)
        layout = FloatLayout(size=Window.size)
        layout.add_widget(self.img1)
        layout.add_widget(self.cap_button)
        M_screen.add_widget(layout)

    def Name_cal(self,instance,value):
        print("name is ", value)
        self.patient.Name = value

    def Age_cal(self,instance,value):
        print("age is ", value)
        self.patient.Age = value

    def Temp_UL_cal(self,instance,value):
        print("Temperature of ulcer is ", value)
        self.patient.Ulcer_temp = value

    def Temp_Bo_cal(self,instance,value):
        print("temperature of body ", value)
        self.patient.Body_temp = value

    def Day_cal(self,instance,value):
        print("date", value)
        self.patient.Day = value

    def pridict_callback(self,instance):
        self.patient.create_new_patient()
        result = self.patient.read_patient()
        print(result)
        self.patient.predict_ulcer_stage(result)

    def data_screen(self, data_screen):
        self.patient = Patient()
        layout = FloatLayout(size=Window.size)
        Name = TextInput(text='Name',pos_hint={'center_x': 0.3, 'center_y': 0.9},size_hint=(0.4, 0.1))
        Name.bind(text=self.Name_cal)

        Age = TextInput(text='Age',pos_hint={'center_x': 0.3, 'center_y': 0.7}, size_hint=(0.4, 0.1))
        Age.bind(text = self.Age_cal)

        Temp_UL = TextInput(text='Temperature of ulcer',pos_hint={'center_x': 0.3, 'center_y': 0.5}, size_hint=(0.4, 0.1))
        Temp_UL.bind(text = self.Temp_UL_cal)

        Temp_Bo = TextInput(text='Temperature of body',pos_hint={'center_x': 0.3, 'center_y': 0.3}, size_hint=(0.4, 0.1))
        Temp_Bo.bind(text=self.Temp_Bo_cal)

        Day =  TextInput(text='Day',pos_hint={'center_x': 0.3, 'center_y': 0.1}, size_hint=(0.4, 0.1))
        Day.bind(text=self.Day_cal)

        self.save_button = Button(text="Save Data", size_hint=(0.2, 0.4),
                                  pos_hint={'center_x': 0.7, 'center_y': 0.4})
        self.save_button.bind(on_press=self.patient.input_info)

        self.predict_button = Button(text="Predict Data", size_hint=(0.2, 0.4),
                                  pos_hint={'center_x': 0.7, 'center_y': 0.7})
        self.predict_button.bind(on_press=self.pridict_callback)


        layout.add_widget(Name)
        layout.add_widget(Age)
        layout.add_widget(Temp_UL)
        layout.add_widget(Temp_Bo)
        layout.add_widget(Day)
        layout.add_widget(self.save_button)
        layout.add_widget(self.predict_button)
        data_screen.add_widget(layout)

    def detection_callback(self,instance):
        self.root.current = 'data_screen'

    def detection_screen(self,Det_scrren):
        layout = FloatLayout(size=Window.size)
        self.size_ratio = None
        self.input_size = (320, 240)
        self.width = self.input_size[0]
        self.height = self.input_size[1]
        self.check_window_size()
        self.img2 = Image(source = "detections.jpeg", pos_hint={'center_x': 0.5, 'center_y': 0.5})
        self.det_button = Button(text="go to data screen", size_hint=(0.2, 0.4),
                                  pos_hint={'center_x': 0.75, 'center_y': 0.125})
        self.det_button.bind(on_press=self.detection_callback)
        layout.add_widget(self.img2)
        layout.add_widget(self.det_button)
        Det_scrren.add_widget(layout)

    def build(self):

        sm = ScreenManager(transition=FadeTransition())
        Welcome_screen = Screen(name = 'welcome_screen')
        Main_Screen = Screen(name= 'main_screen')
        Data_screen = Screen(name= 'data_screen')
        Detection_screen = Screen(name= "detection_screen")

        sm.add_widget(Welcome_screen)
        sm.add_widget(Main_Screen)
        sm.add_widget(Data_screen)
        sm.add_widget(Detection_screen)

        self.welcome_screen(Welcome_screen)
        self.main_screen(Main_Screen)
        self.data_screen(Data_screen)
        self.detection_screen(Detection_screen)

        self.display_speed = 0  # 0 for best resolution, 1 for medium, 2 for fastest display
        # desired_resolution = (720, 480)
        self.camCV = CameraCV(play=True)
        self.camCV.image_bytes = False

        Clock.schedule_interval(self.update_texture, 1.0 / 60.0)
        # Clock.schedule_interval(self.update_texture2, 1.0 / 60.0)

        return sm

    def set_display_speed(self, instance):
        if self.display_speed == 2:
            self.display_speed = 0
        else:
            self.display_speed += 1

    def check_window_size(self):
        self.window_shape = Window.size
        self.window_width = self.window_shape[0]
        self.window_height = self.window_shape[1]
        Logger.info(f"Screen: Window size is {self.window_shape}")

    def update_texture(self, instance):
        self.check_window_size()
        if type(self.camCV.image_bytes) == bool:
            Logger.info("Camera: No valid frame")
            return
        Logger.info(f"Camera: image bytes {len(self.camCV.image_bytes)}")
        Logger.info(f"Camera: image size {self.camCV.image_size}")
        if not self.size_ratio:
            self.camera_width = self.camCV.image_size[0]
            self.camera_height = self.camCV.image_size[1]
            self.size_ratio = self.camera_height / self.camera_width

        self.extract_frame()
        self.process_frame()
        self.display_frame()
        Logger.info(f"Camera: converted to gray and back to rgba")

    def update_texture2(self, instance):
        self.check_window_size()
        if type(self.camCV.image_bytes) == bool:
            Logger.info("Camera: No valid frame")
            return
        Logger.info(f"Camera: image bytes {len(self.camCV.image_bytes)}")
        Logger.info(f"Camera: image size {self.camCV.image_size}")
        if not self.size_ratio:
            self.camera_width = self.camCV.image_size[0]
            self.camera_height = self.camCV.image_size[1]
            self.size_ratio = self.camera_height / self.camera_width

        self.extract_frame()
        self.process_frame()
        self.display_frame2()
        Logger.info(f"Camera: converted to gray and back to rgba")

    def extract_frame(self):
        self.frame = np.frombuffer(self.camCV.image_bytes, np.uint8)
        Logger.info(f"Camera: frame exist")
        self.frame = self.frame.reshape((self.camCV.image_size[1], self.camCV.image_size[0], 4))
        # self.imggg = self.frame
        self.imggg = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        Logger.info(f"Camera: frame size {self.frame.shape}")

    def process_frame(self):
        self.frame = cv2.rotate(self.frame, cv2.ROTATE_180)

    def make_new_texture_frame(self):
        Logger.info(f"Camera: Displaying frame")
        self.frame = cv2.resize(self.frame, (int(self.window_height * self.size_ratio), self.window_height))
        self.frame = self.frame.reshape((self.frame.shape[1], self.frame.shape[0], 4))
        buf = self.frame.tostring()
        Logger.info(f"Camera: converted to bytes {len(buf)}")
        texture1 = Texture.create(size=(self.frame.shape[0], self.frame.shape[1]), colorfmt='rgba')
        texture1.blit_buffer(buf, colorfmt='rgba', bufferfmt='ubyte')
        return texture1

    def display_frame(self):
        self.img1.texture = self.make_new_texture_frame()

    def display_frame2(self):
        self.img2.texture = self.make_new_texture_frame()

if __name__ == '__main__':
    CamApp().run()
