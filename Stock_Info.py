#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# generated by wxGlade 0.9.3 on Mon Jun 10 00:10:03 2019
#

from nsetools import Nse

import wx

# begin wxGlade: dependencies
# end wxGlade

# begin wxGlade: extracode
# end wxGlade


class Stk_Info_MyFrame(wx.Frame):
    def __init__(self, *args, **kwds):
        # begin wxGlade: Stk_Info_MyFrame.__init__
        kwds["style"] = kwds.get("style", 0) | wx.DEFAULT_FRAME_STYLE
        wx.Frame.__init__(self, *args, **kwds)
        self.SetSize((400, 300))

        self.__set_properties()
        self.__do_layout()
        # end wxGlade

    def __set_properties(self):
        # begin wxGlade: Stk_Info_MyFrame.__set_properties
        self.SetTitle("frame")
        # end wxGlade

    def __do_layout(self):
        # begin wxGlade: Stk_Info_MyFrame.__do_layout
        sizer_1 = wx.BoxSizer(wx.VERTICAL)
        sizer_19 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_21 = wx.BoxSizer(wx.VERTICAL)
        sizer_20 = wx.BoxSizer(wx.VERTICAL)
        sizer_16 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_18 = wx.BoxSizer(wx.VERTICAL)
        sizer_17 = wx.BoxSizer(wx.VERTICAL)
        sizer_13 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_15 = wx.BoxSizer(wx.VERTICAL)
        sizer_14 = wx.BoxSizer(wx.VERTICAL)
        sizer_7 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_12 = wx.BoxSizer(wx.VERTICAL)
        sizer_11 = wx.BoxSizer(wx.VERTICAL)
        sizer_10 = wx.BoxSizer(wx.VERTICAL)
        sizer_9 = wx.BoxSizer(wx.VERTICAL)
        sizer_8 = wx.BoxSizer(wx.VERTICAL)
        sizer_2 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_6 = wx.BoxSizer(wx.VERTICAL)
        sizer_5 = wx.BoxSizer(wx.VERTICAL)
        sizer_4 = wx.BoxSizer(wx.VERTICAL)
        sizer_3 = wx.BoxSizer(wx.VERTICAL)
        static_line_6 = wx.StaticLine(self, wx.ID_ANY)
        sizer_1.Add(static_line_6, 0, wx.ALL | wx.EXPAND, 5)
        static_line_7 = wx.StaticLine(self, wx.ID_ANY)
        sizer_1.Add(static_line_7, 0, wx.ALL | wx.EXPAND, 3)
        label_1 = wx.StaticText(self, wx.ID_ANY, "STOCK INFORMATION")
        label_1.SetFont(wx.Font(20, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 1, ""))
        sizer_1.Add(label_1, 0, wx.ALIGN_CENTER, 0)
        static_line_8 = wx.StaticLine(self, wx.ID_ANY)
        sizer_1.Add(static_line_8, 0, wx.ALL | wx.EXPAND, 5)
        static_line_9 = wx.StaticLine(self, wx.ID_ANY)
        sizer_1.Add(static_line_9, 0, wx.ALL | wx.EXPAND, 3)
        label_2 = wx.StaticText(self, wx.ID_ANY, "\nSTOCK ")
        label_2.SetFont(wx.Font(15, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 1, ""))
        sizer_1.Add(label_2, 0, 0, 0)
        label_42 = wx.StaticText(self, wx.ID_ANY, stock_Info['symbol']+' - '+stock_Info['companyName']+' - '+stock_Info['series'])
        label_42.SetFont(wx.Font(15, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_1.Add(label_42, 0, 0, 0)
        static_line_10 = wx.StaticLine(self, wx.ID_ANY)
        sizer_1.Add(static_line_10, 0, wx.ALL | wx.EXPAND, 3)
        label_19 = wx.StaticText(self, wx.ID_ANY, "\nORDER BOOK")
        label_19.SetFont(wx.Font(15, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 1, ""))
        sizer_1.Add(label_19, 0, 0, 0)
        label_3 = wx.StaticText(self, wx.ID_ANY, "BUY PRICE")
        label_3.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_3.Add(label_3, 0, wx.ALIGN_CENTER, 0)
        label_4 = wx.StaticText(self, wx.ID_ANY, str(stock_Info['buyPrice1']))
        label_4.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_3.Add(label_4, 0, wx.ALIGN_CENTER, 0)
        label_5 = wx.StaticText(self, wx.ID_ANY, str(stock_Info['buyPrice2']))
        label_5.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_3.Add(label_5, 0, wx.ALIGN_CENTER, 0)
        label_6 = wx.StaticText(self, wx.ID_ANY, str(stock_Info['buyPrice3']))
        label_6.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_3.Add(label_6, 0, wx.ALIGN_CENTER, 0)
        sizer_2.Add(sizer_3, 1, wx.EXPAND, 0)
        label_7 = wx.StaticText(self, wx.ID_ANY, "BUY QUANTITY")
        label_7.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_4.Add(label_7, 0, wx.ALIGN_CENTER, 0)
        label_8 = wx.StaticText(self, wx.ID_ANY, str(stock_Info['buyQuantity1']))
        label_8.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_4.Add(label_8, 0, wx.ALIGN_CENTER, 0)
        label_9 = wx.StaticText(self, wx.ID_ANY, str(stock_Info['buyQuantity2']))
        label_9.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_4.Add(label_9, 0, wx.ALIGN_CENTER, 0)
        label_10 = wx.StaticText(self, wx.ID_ANY, str(stock_Info['buyQuantity3']))
        label_10.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_4.Add(label_10, 0, wx.ALIGN_CENTER, 0)
        sizer_2.Add(sizer_4, 1, wx.EXPAND, 0)
        label_11 = wx.StaticText(self, wx.ID_ANY, "SELL PRICE")
        label_11.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_5.Add(label_11, 0, wx.ALIGN_CENTER, 0)
        label_12 = wx.StaticText(self, wx.ID_ANY, str(stock_Info['sellPrice1']))
        label_12.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_5.Add(label_12, 0, wx.ALIGN_CENTER, 0)
        label_13 = wx.StaticText(self, wx.ID_ANY, str(stock_Info['sellPrice2']))
        label_13.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_5.Add(label_13, 0, wx.ALIGN_CENTER, 0)
        label_14 = wx.StaticText(self, wx.ID_ANY, str(stock_Info['sellPrice3']))
        label_14.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_5.Add(label_14, 0, wx.ALIGN_CENTER, 0)
        sizer_2.Add(sizer_5, 1, wx.EXPAND, 0)
        label_15 = wx.StaticText(self, wx.ID_ANY, "SELL QUANTITY")
        label_15.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_6.Add(label_15, 0, wx.ALIGN_CENTER, 0)
        label_16 = wx.StaticText(self, wx.ID_ANY, str(stock_Info['sellQuantity1']))
        label_16.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_6.Add(label_16, 0, wx.ALIGN_CENTER, 0)
        label_17 = wx.StaticText(self, wx.ID_ANY, str(stock_Info['sellQuantity2']))
        label_17.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_6.Add(label_17, 0, wx.ALIGN_CENTER, 0)
        label_18 = wx.StaticText(self, wx.ID_ANY, str(stock_Info['sellQuantity3']))
        label_18.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_6.Add(label_18, 0, wx.ALIGN_CENTER, 0)
        sizer_2.Add(sizer_6, 1, wx.EXPAND, 0)
        sizer_1.Add(sizer_2, 1, wx.EXPAND, 0)
        static_line_11 = wx.StaticLine(self, wx.ID_ANY)
        sizer_1.Add(static_line_11, 0, wx.ALL | wx.EXPAND, 5)
        label_20 = wx.StaticText(self, wx.ID_ANY, "\nOPEN")
        label_20.SetFont(wx.Font(15, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_8.Add(label_20, 0, wx.ALIGN_CENTER, 0)
        label_21 = wx.StaticText(self, wx.ID_ANY, str(stock_Info['open']))
        label_21.SetFont(wx.Font(15, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_8.Add(label_21, 0, wx.ALIGN_CENTER, 0)
        sizer_7.Add(sizer_8, 1, wx.EXPAND, 0)
        label_22 = wx.StaticText(self, wx.ID_ANY, "\nHIGH")
        label_22.SetFont(wx.Font(15, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_9.Add(label_22, 0, wx.ALIGN_CENTER, 0)
        label_23 = wx.StaticText(self, wx.ID_ANY, str(stock_Info['dayHigh']))
        label_23.SetFont(wx.Font(15, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_9.Add(label_23, 0, wx.ALIGN_CENTER, 0)
        sizer_7.Add(sizer_9, 1, wx.EXPAND, 0)
        label_24 = wx.StaticText(self, wx.ID_ANY, "\nLOW")
        label_24.SetFont(wx.Font(15, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_10.Add(label_24, 0, wx.ALIGN_CENTER, 0)
        label_25 = wx.StaticText(self, wx.ID_ANY, str(stock_Info['dayLow']))
        label_25.SetFont(wx.Font(15, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_10.Add(label_25, 0, wx.ALIGN_CENTER, 0)
        sizer_7.Add(sizer_10, 1, wx.EXPAND, 0)
        label_26 = wx.StaticText(self, wx.ID_ANY, "\nCLOSE")
        label_26.SetFont(wx.Font(15, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_11.Add(label_26, 0, wx.ALIGN_CENTER, 0)
        if stock_Info['closePrice']==0:
            label_27 = wx.StaticText(self, wx.ID_ANY, str(stock_Info['lastPrice']))
        else:
            label_27 = wx.StaticText(self, wx.ID_ANY, str(stock_Info['closePrice']))
        label_27.SetFont(wx.Font(15, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_11.Add(label_27, 0, wx.ALIGN_CENTER, 0)
        sizer_7.Add(sizer_11, 1, wx.EXPAND, 0)
        label_28 = wx.StaticText(self, wx.ID_ANY, "\nLAST")
        label_28.SetFont(wx.Font(15, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_12.Add(label_28, 0, wx.ALIGN_CENTER, 0)
        label_29 = wx.StaticText(self, wx.ID_ANY, str(stock_Info['lastPrice']))
        label_29.SetFont(wx.Font(15, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_12.Add(label_29, 0, wx.ALIGN_CENTER, 0)
        sizer_7.Add(sizer_12, 1, wx.EXPAND, 0)
        sizer_1.Add(sizer_7, 1, wx.EXPAND, 0)
        static_line_12 = wx.StaticLine(self, wx.ID_ANY)
        sizer_1.Add(static_line_12, 0, wx.ALL | wx.EXPAND, 5)
        label_30 = wx.StaticText(self, wx.ID_ANY, "\n52 WEEKS HIGH")
        label_30.SetFont(wx.Font(15, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_14.Add(label_30, 0, wx.ALIGN_CENTER, 0)
        label_31 = wx.StaticText(self, wx.ID_ANY, str(stock_Info['high52']))
        label_31.SetFont(wx.Font(15, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_14.Add(label_31, 0, wx.ALIGN_CENTER, 0)
        sizer_13.Add(sizer_14, 1, wx.EXPAND, 0)
        label_32 = wx.StaticText(self, wx.ID_ANY, "\n52 WEEKS LOW")
        label_32.SetFont(wx.Font(15, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_15.Add(label_32, 0, wx.ALIGN_CENTER, 0)
        label_33 = wx.StaticText(self, wx.ID_ANY, str(stock_Info['low52']))
        label_33.SetFont(wx.Font(15, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_15.Add(label_33, 0, wx.ALIGN_CENTER, 0)
        sizer_13.Add(sizer_15, 1, wx.EXPAND, 0)
        sizer_1.Add(sizer_13, 1, wx.EXPAND, 0)
        static_line_13 = wx.StaticLine(self, wx.ID_ANY)
        sizer_1.Add(static_line_13, 0, wx.ALL | wx.EXPAND, 5)
        label_34 = wx.StaticText(self, wx.ID_ANY, "\nTOTAL BUY QUANTITY")
        label_34.SetFont(wx.Font(15, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_17.Add(label_34, 0, wx.ALIGN_CENTER, 0)
        label_35 = wx.StaticText(self, wx.ID_ANY, str(stock_Info['totalBuyQuantity']))
        label_35.SetFont(wx.Font(15, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_17.Add(label_35, 0, wx.ALIGN_CENTER, 0)
        sizer_16.Add(sizer_17, 1, wx.EXPAND, 0)
        label_36 = wx.StaticText(self, wx.ID_ANY, "\nTOTAL SELL QUANTITY")
        label_36.SetFont(wx.Font(15, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_18.Add(label_36, 0, wx.ALIGN_CENTER, 0)
        label_37 = wx.StaticText(self, wx.ID_ANY, str(stock_Info['totalSellQuantity']))
        label_37.SetFont(wx.Font(15, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_18.Add(label_37, 0, wx.ALIGN_CENTER, 0)
        sizer_16.Add(sizer_18, 1, wx.EXPAND, 0)
        sizer_1.Add(sizer_16, 1, wx.EXPAND, 0)
        static_line_14 = wx.StaticLine(self, wx.ID_ANY)
        sizer_1.Add(static_line_14, 0, wx.ALL | wx.EXPAND, 5)
        label_38 = wx.StaticText(self, wx.ID_ANY, "\nTOTAL TRADED VALUE")
        label_38.SetFont(wx.Font(15, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_20.Add(label_38, 0, wx.ALIGN_CENTER, 0)
        label_39 = wx.StaticText(self, wx.ID_ANY, str(stock_Info['totalTradedValue']))
        label_39.SetFont(wx.Font(15, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_20.Add(label_39, 0, wx.ALIGN_CENTER, 0)
        sizer_19.Add(sizer_20, 1, wx.EXPAND, 0)
        label_40 = wx.StaticText(self, wx.ID_ANY, "\nTOTAL TRADED VOLUME")
        label_40.SetFont(wx.Font(15, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_21.Add(label_40, 0, wx.ALIGN_CENTER, 0)
        label_41 = wx.StaticText(self, wx.ID_ANY, str(stock_Info['totalTradedVolume']))
        label_41.SetFont(wx.Font(15, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_21.Add(label_41, 0, wx.ALIGN_CENTER, 0)
        sizer_19.Add(sizer_21, 1, wx.EXPAND, 0)
        sizer_1.Add(sizer_19, 1, wx.EXPAND, 0)
        static_line_15 = wx.StaticLine(self, wx.ID_ANY)
        sizer_1.Add(static_line_15, 0, wx.ALL | wx.EXPAND, 5)
        self.SetSizer(sizer_1)
        self.Layout()
        # end wxGlade

# end of class Stk_Info_MyFrame

class Stk_Info_MyApp(wx.App):
    def OnInit(self):
        self.Stk_info_frame = Stk_Info_MyFrame(None, wx.ID_ANY, "")
        self.SetTopWindow(self.Stk_info_frame)
        self.Stk_info_frame.Show()
        return True

# end of class Stk_Info_MyApp

stock_Info={}

def Stock_info(Stock):
    global stock_Info
    nse=Nse()
    stock_Info=nse.get_quote(Stock)
    Stk_Info_app = Stk_Info_MyApp(0)
    Stk_Info_app.MainLoop()
