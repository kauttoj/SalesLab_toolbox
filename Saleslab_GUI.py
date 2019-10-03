# start wxPython GUI to enter and process files
import time
import wx
from threading import Thread
from wx.lib.pubsub import pub
import Saleslab_settings

global inputfile_list
global is_running

def start_GUI(GUI_title_str = "Saleslab"):

    class TestThread(Thread):
        """Test Worker Thread Class."""
        # ----------------------------------------------------------------------
        def __init__(self,inputfile_list):
            """Init Worker Thread Class."""
            Thread.__init__(self)
            self.inputfile_list=inputfile_list
            self.start()  # start the thread

        # ----------------------------------------------------------------------
        def run(self):
            """Run Worker Thread."""
            # This is the code executing in the new thread.
            pub.sendMessage("panel_listener", message="running")
            for f in self.inputfile_list:
                try:
                    Saleslab_settings.INPUT_FILE = f
                    Saleslab_settings.process_file()
                except BaseException as error:
                    Saleslab_settings.myprint('\nERROR in processing!! {}: {}'.format(type(error).__name__,str(error)))
            pub.sendMessage("panel_listener", message="stopped")
        ########################################################################
    class MyFileDropTarget(wx.FileDropTarget):
        """"""
        # ----------------------------------------------------------------------
        def __init__(self, window):
            """Constructor"""
            wx.FileDropTarget.__init__(self)
            self.window = window

        # ----------------------------------------------------------------------
        def OnDropFiles(self, x, y, filenames):
            """
            When files are dropped, write where they were dropped and then
            the file paths themselves
            """
            global inputfile_list

            if not is_running:
                self.window.SetInsertionPointEnd()
                self.window.updateText("INPUT FILE (%i):\n\"%s\"\n" % (len(filenames),"\n".join(filenames)),clear=True)
                inputfile_list = filenames
                return True
            #for filepath in filenames:
            #    self.window.updateText(filepath + '\n')
            return False
                ########################################################################

    class DnDPanel(wx.Panel):
        """"""
        # ----------------------------------------------------------------------
        def __init__(self, parent):
            """Constructor"""
            wx.Panel.__init__(self, parent=parent)

            file_drop_target = MyFileDropTarget(self)
            lbl = wx.StaticText(self, label="Drag one or more raw iMotions datafiles here:")
            self.fileTextCtrl = wx.TextCtrl(self,style=wx.TE_READONLY | wx.TE_MULTILINE) # wx.HSCROLL
            self.fileTextCtrl.SetDropTarget(file_drop_target)

            sizer = wx.BoxSizer(wx.VERTICAL)
            sizer.Add(lbl, 0, wx.ALL, 5)
            sizer.Add(self.fileTextCtrl, 1, wx.EXPAND | wx.ALL, 5)

            self.button = wx.Button(self, label="Process file")

            sizer.Add(self.button,0.5, wx.EXPAND | wx.ALL)
            self.button.Bind(wx.EVT_BUTTON, self.OnClicked)

            self.SetSizer(sizer)

            pub.subscribe(self.my_listener, "panel_listener")

            Saleslab_settings.updateTextFun = self.updateText

        # ----------------------------------------------------------------------
        def my_listener(self, message, arg2=None):
            #print(f"Received the following message: {message}")
            global is_running
            if message == "running":
                self.updateText("\n==Processing started (%s)==\n" % time.strftime('%H:%M:%S'))
                is_running=True
                self.button.Disable()
            if message == "stopped":
                is_running = False
                self.button.Enable()
                self.updateText("\n==Processing finished (%s)==\n" % time.strftime('%H:%M:%S'))

        def SetInsertionPointEnd(self):
            """
            Put insertion point at end of text control to prevent overwriting
            """
            self.fileTextCtrl.SetInsertionPointEnd()

        # ----------------------------------------------------------------------
        def updateText(self,text,clear=False):
            """
            Write text to the text control
            """
            if clear is True:
                self.fileTextCtrl.Clear()
            self.fileTextCtrl.AppendText(text)

        def OnClicked(self, event):
            TestThread(inputfile_list)
            #btn = event.GetEventObject()
            #btn.Disable()

            #print("Label of pressed button = ", btn)

    ########################################################################
    class DnDFrame(wx.Frame):
        """"""
        # ----------------------------------------------------------------------
        def __init__(self):
            """Constructor"""
            wx.Frame.__init__(self, parent=None, title=GUI_title_str,pos=(30,30), size=(500,550))
            panel = DnDPanel(self)
            self.Show()

    # ----------------------------------------------------------------------
    global inputfile_list
    global is_running

    inputfile_list = []
    is_running = False

    app = wx.App(False)
    frame = DnDFrame()
    app.MainLoop()