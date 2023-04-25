import wx

class MyFrame(wx.Frame):
    def __init__(self, parent, title):
        super(MyFrame, self).__init__(parent, title=title, size=(300, 200))
        self.InitUI()

    def InitUI(self):
        # 创建一个面板
        pnl = wx.Panel(self)
        # 创建一个按钮
        self.btn = wx.Button(pnl, label='Click me!', pos=(100, 50))
        # 绑定按钮事件
        self.btn.Bind(wx.EVT_BUTTON, self.OnClick)

    def OnClick(self, e):
        # 设置按钮的标签
        self.btn.SetLabel("Clicked")

if __name__ == '__main__':
    app = wx.App()
    frame = MyFrame(None, title='Hello world')
    frame.Show()
    app.MainLoop()
