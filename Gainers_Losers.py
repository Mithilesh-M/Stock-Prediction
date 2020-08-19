#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# generated by wxGlade 0.9.3 on Sun Jun  9 23:39:17 2019
#

from nsetools import Nse

import wx


# begin wxGlade: dependencies
# end wxGlade

# begin wxGlade: extracode
# end wxGlade


class MyFrame_Gainers_Losers(wx.Frame):
    def __init__(self, *args, **kwds):
        # begin wxGlade: MyFrame_Gainers.__init__
        kwds["style"] = kwds.get("style", 0) | wx.DEFAULT_FRAME_STYLE
        wx.Frame.__init__(self, *args, **kwds)
        self.SetSize((400, 300))

        self.__set_properties()
        self.__do_layout()
        # end wxGlade

    def __set_properties(self):
        # begin wxGlade: MyFrame_Gainers.__set_properties
        self.SetTitle("frame")
        # end wxGlade

    def __do_layout(self):
        # begin wxGlade: MyFrame_Gainers.__do_layout
        sizer_1 = wx.BoxSizer(wx.VERTICAL)
        sizer_30 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_36 = wx.BoxSizer(wx.VERTICAL)
        sizer_35 = wx.BoxSizer(wx.VERTICAL)
        sizer_34 = wx.BoxSizer(wx.VERTICAL)
        sizer_33 = wx.BoxSizer(wx.VERTICAL)
        sizer_32 = wx.BoxSizer(wx.VERTICAL)
        sizer_31 = wx.BoxSizer(wx.VERTICAL)
        sizer_23 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_29 = wx.BoxSizer(wx.VERTICAL)
        sizer_28 = wx.BoxSizer(wx.VERTICAL)
        sizer_27 = wx.BoxSizer(wx.VERTICAL)
        sizer_26 = wx.BoxSizer(wx.VERTICAL)
        sizer_25 = wx.BoxSizer(wx.VERTICAL)
        sizer_24 = wx.BoxSizer(wx.VERTICAL)
        sizer_16 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_22 = wx.BoxSizer(wx.VERTICAL)
        sizer_21 = wx.BoxSizer(wx.VERTICAL)
        sizer_20 = wx.BoxSizer(wx.VERTICAL)
        sizer_19 = wx.BoxSizer(wx.VERTICAL)
        sizer_18 = wx.BoxSizer(wx.VERTICAL)
        sizer_17 = wx.BoxSizer(wx.VERTICAL)
        sizer_9 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_15 = wx.BoxSizer(wx.VERTICAL)
        sizer_14 = wx.BoxSizer(wx.VERTICAL)
        sizer_13 = wx.BoxSizer(wx.VERTICAL)
        sizer_12 = wx.BoxSizer(wx.VERTICAL)
        sizer_11 = wx.BoxSizer(wx.VERTICAL)
        sizer_10 = wx.BoxSizer(wx.VERTICAL)
        sizer_2 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_8 = wx.BoxSizer(wx.VERTICAL)
        sizer_7 = wx.BoxSizer(wx.VERTICAL)
        sizer_6 = wx.BoxSizer(wx.VERTICAL)
        sizer_5 = wx.BoxSizer(wx.VERTICAL)
        sizer_4 = wx.BoxSizer(wx.VERTICAL)
        sizer_3 = wx.BoxSizer(wx.VERTICAL)
        static_line_6 = wx.StaticLine(self, wx.ID_ANY)
        sizer_1.Add(static_line_6, 0, wx.ALL | wx.EXPAND, 5)
        static_line_8 = wx.StaticLine(self, wx.ID_ANY)
        sizer_1.Add(static_line_8, 0, wx.ALL | wx.EXPAND, 3)
        if Selection=='Gain':
            label_66 = wx.StaticText(self, wx.ID_ANY, "\nTOP STOCK GAINERS\n")
        else:
            label_66 = wx.StaticText(self, wx.ID_ANY, "\nTOP STOCK LOSERS\n")
        label_66.SetFont(wx.Font(20, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 1, ""))
        sizer_1.Add(label_66, 0, wx.ALIGN_CENTER, 0)
        static_line_1 = wx.StaticLine(self, wx.ID_ANY)
        sizer_1.Add(static_line_1, 0, wx.ALL | wx.EXPAND, 5)
        static_line_9 = wx.StaticLine(self, wx.ID_ANY)
        sizer_1.Add(static_line_9, 0, wx.ALL | wx.EXPAND, 3)
        label_1 = wx.StaticText(self, wx.ID_ANY, '\n'+Top_Stock[0]['symbol']+'\n')
        label_1.SetFont(wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 1, ""))
        sizer_1.Add(label_1, 0, 0, 0)
        label_2 = wx.StaticText(self, wx.ID_ANY, "OPEN")
        label_2.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_3.Add(label_2, 0, wx.ALIGN_CENTER, 0)
        label_3 = wx.StaticText(self, wx.ID_ANY, str(Top_Stock[0]['openPrice']))
        label_3.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_3.Add(label_3, 0, wx.ALIGN_CENTER, 0)
        sizer_2.Add(sizer_3, 1, wx.EXPAND, 0)
        label_4 = wx.StaticText(self, wx.ID_ANY, "HIGH")
        label_4.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_4.Add(label_4, 0, wx.ALIGN_CENTER, 0)
        label_5 = wx.StaticText(self, wx.ID_ANY, str(Top_Stock[0]['highPrice']))
        label_5.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_4.Add(label_5, 0, wx.ALIGN_CENTER, 0)
        sizer_2.Add(sizer_4, 1, wx.EXPAND, 0)
        label_6 = wx.StaticText(self, wx.ID_ANY, "LOW")
        label_6.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_5.Add(label_6, 0, wx.ALIGN_CENTER, 0)
        label_7 = wx.StaticText(self, wx.ID_ANY, str(Top_Stock[0]['lowPrice']))
        label_7.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_5.Add(label_7, 0, wx.ALIGN_CENTER, 0)
        sizer_2.Add(sizer_5, 1, wx.EXPAND, 0)
        label_8 = wx.StaticText(self, wx.ID_ANY, "CLOSE")
        label_8.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_6.Add(label_8, 0, wx.ALIGN_CENTER, 0)
        label_9 = wx.StaticText(self, wx.ID_ANY, str(Top_Stock[0]['ltp']))
        label_9.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_6.Add(label_9, 0, wx.ALIGN_CENTER, 0)
        sizer_2.Add(sizer_6, 1, wx.EXPAND, 0)
        label_10 = wx.StaticText(self, wx.ID_ANY, "VOLUME")
        label_10.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_7.Add(label_10, 0, wx.ALIGN_CENTER, 0)
        label_11 = wx.StaticText(self, wx.ID_ANY, str(Top_Stock[0]['tradedQuantity']))
        label_11.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_7.Add(label_11, 0, wx.ALIGN_CENTER, 0)
        sizer_2.Add(sizer_7, 1, wx.EXPAND, 0)
        label_12 = wx.StaticText(self, wx.ID_ANY, "TURNOVER")
        label_12.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_8.Add(label_12, 0, wx.ALIGN_CENTER, 0)
        label_13 = wx.StaticText(self, wx.ID_ANY, str(Top_Stock[0]['turnoverInLakhs']))
        label_13.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_8.Add(label_13, 0, wx.ALIGN_CENTER, 0)
        sizer_2.Add(sizer_8, 1, wx.EXPAND, 0)
        sizer_1.Add(sizer_2, 1, wx.EXPAND, 0)
        static_line_2 = wx.StaticLine(self, wx.ID_ANY)
        sizer_1.Add(static_line_2, 0, wx.ALL | wx.EXPAND, 5)
        label_14 = wx.StaticText(self, wx.ID_ANY, '\n'+Top_Stock[1]['symbol']+'\n')
        label_14.SetFont(wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 1, ""))
        sizer_1.Add(label_14, 0, 0, 0)
        label_15 = wx.StaticText(self, wx.ID_ANY, "OPEN")
        label_15.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_10.Add(label_15, 0, wx.ALIGN_CENTER, 0)
        label_16 = wx.StaticText(self, wx.ID_ANY, str(Top_Stock[1]['openPrice']))
        label_16.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_10.Add(label_16, 0, wx.ALIGN_CENTER, 0)
        sizer_9.Add(sizer_10, 1, wx.EXPAND, 0)
        label_17 = wx.StaticText(self, wx.ID_ANY, "HIGH")
        label_17.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_11.Add(label_17, 0, wx.ALIGN_CENTER, 0)
        label_18 = wx.StaticText(self, wx.ID_ANY, str(Top_Stock[1]['highPrice']))
        label_18.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_11.Add(label_18, 0, wx.ALIGN_CENTER, 0)
        sizer_9.Add(sizer_11, 1, wx.EXPAND, 0)
        label_19 = wx.StaticText(self, wx.ID_ANY, "LOW")
        label_19.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_12.Add(label_19, 0, wx.ALIGN_CENTER, 0)
        label_20 = wx.StaticText(self, wx.ID_ANY, str(Top_Stock[1]['lowPrice']))
        label_20.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_12.Add(label_20, 0, wx.ALIGN_CENTER, 0)
        sizer_9.Add(sizer_12, 1, wx.EXPAND, 0)
        label_21 = wx.StaticText(self, wx.ID_ANY, "CLOSE")
        label_21.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_13.Add(label_21, 0, wx.ALIGN_CENTER, 0)
        label_22 = wx.StaticText(self, wx.ID_ANY, str(Top_Stock[1]['ltp']))
        label_22.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_13.Add(label_22, 0, wx.ALIGN_CENTER, 0)
        sizer_9.Add(sizer_13, 1, wx.EXPAND, 0)
        label_23 = wx.StaticText(self, wx.ID_ANY, "VOLUME")
        label_23.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_14.Add(label_23, 0, wx.ALIGN_CENTER, 0)
        label_24 = wx.StaticText(self, wx.ID_ANY, str(Top_Stock[1]['tradedQuantity']))
        label_24.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_14.Add(label_24, 0, wx.ALIGN_CENTER, 0)
        sizer_9.Add(sizer_14, 1, wx.EXPAND, 0)
        label_25 = wx.StaticText(self, wx.ID_ANY, "TURNOVER")
        label_25.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_15.Add(label_25, 0, wx.ALIGN_CENTER, 0)
        label_26 = wx.StaticText(self, wx.ID_ANY, str(Top_Stock[1]['turnoverInLakhs']))
        label_26.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_15.Add(label_26, 0, wx.ALIGN_CENTER, 0)
        sizer_9.Add(sizer_15, 1, wx.EXPAND, 0)
        sizer_1.Add(sizer_9, 1, wx.EXPAND, 0)
        static_line_3 = wx.StaticLine(self, wx.ID_ANY)
        sizer_1.Add(static_line_3, 0, wx.ALL | wx.EXPAND, 5)
        label_27 = wx.StaticText(self, wx.ID_ANY, '\n'+Top_Stock[2]['symbol']+'\n')
        label_27.SetFont(wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 1, ""))
        sizer_1.Add(label_27, 0, 0, 0)
        label_28 = wx.StaticText(self, wx.ID_ANY, "OPEN")
        label_28.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_17.Add(label_28, 0, wx.ALIGN_CENTER, 0)
        label_29 = wx.StaticText(self, wx.ID_ANY, str(Top_Stock[2]['openPrice']))
        label_29.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_17.Add(label_29, 0, wx.ALIGN_CENTER, 0)
        sizer_16.Add(sizer_17, 1, wx.EXPAND, 0)
        label_30 = wx.StaticText(self, wx.ID_ANY, "HIGH")
        label_30.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_18.Add(label_30, 0, wx.ALIGN_CENTER, 0)
        label_31 = wx.StaticText(self, wx.ID_ANY, str(Top_Stock[2]['highPrice']))
        label_31.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_18.Add(label_31, 0, wx.ALIGN_CENTER, 0)
        sizer_16.Add(sizer_18, 1, wx.EXPAND, 0)
        label_32 = wx.StaticText(self, wx.ID_ANY, "LOW")
        label_32.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_19.Add(label_32, 0, wx.ALIGN_CENTER, 0)
        label_33 = wx.StaticText(self, wx.ID_ANY, str(Top_Stock[2]['lowPrice']))
        label_33.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_19.Add(label_33, 0, wx.ALIGN_CENTER, 0)
        sizer_16.Add(sizer_19, 1, wx.EXPAND, 0)
        label_34 = wx.StaticText(self, wx.ID_ANY, "CLOSE")
        label_34.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_20.Add(label_34, 0, wx.ALIGN_CENTER, 0)
        label_35 = wx.StaticText(self, wx.ID_ANY, str(Top_Stock[2]['ltp']))
        label_35.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_20.Add(label_35, 0, wx.ALIGN_CENTER, 0)
        sizer_16.Add(sizer_20, 1, wx.EXPAND, 0)
        label_36 = wx.StaticText(self, wx.ID_ANY, "VOLUME")
        label_36.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_21.Add(label_36, 0, wx.ALIGN_CENTER, 0)
        label_37 = wx.StaticText(self, wx.ID_ANY, str(Top_Stock[2]['tradedQuantity']))
        label_37.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_21.Add(label_37, 0, wx.ALIGN_CENTER, 0)
        sizer_16.Add(sizer_21, 1, wx.EXPAND, 0)
        label_38 = wx.StaticText(self, wx.ID_ANY, "TURNOVER")
        label_38.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_22.Add(label_38, 0, wx.ALIGN_CENTER, 0)
        label_39 = wx.StaticText(self, wx.ID_ANY, str(Top_Stock[2]['turnoverInLakhs']))
        label_39.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_22.Add(label_39, 0, wx.ALIGN_CENTER, 0)
        sizer_16.Add(sizer_22, 1, wx.EXPAND, 0)
        sizer_1.Add(sizer_16, 1, wx.EXPAND, 0)
        static_line_4 = wx.StaticLine(self, wx.ID_ANY)
        sizer_1.Add(static_line_4, 0, wx.ALL | wx.EXPAND, 5)
        label_40 = wx.StaticText(self, wx.ID_ANY, '\n'+Top_Stock[3]['symbol']+'\n')
        label_40.SetFont(wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 1, ""))
        sizer_1.Add(label_40, 0, 0, 0)
        label_41 = wx.StaticText(self, wx.ID_ANY, "OPEN")
        label_41.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_24.Add(label_41, 0, wx.ALIGN_CENTER, 0)
        label_42 = wx.StaticText(self, wx.ID_ANY, str(Top_Stock[3]['openPrice']))
        label_42.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_24.Add(label_42, 0, wx.ALIGN_CENTER, 0)
        sizer_23.Add(sizer_24, 1, wx.EXPAND, 0)
        label_43 = wx.StaticText(self, wx.ID_ANY, "HIGH")
        label_43.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_25.Add(label_43, 0, wx.ALIGN_CENTER, 0)
        label_44 = wx.StaticText(self, wx.ID_ANY, str(Top_Stock[3]['highPrice']))
        label_44.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_25.Add(label_44, 0, wx.ALIGN_CENTER, 0)
        sizer_23.Add(sizer_25, 1, wx.EXPAND, 0)
        label_45 = wx.StaticText(self, wx.ID_ANY, "LOW")
        label_45.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_26.Add(label_45, 0, wx.ALIGN_CENTER, 0)
        label_46 = wx.StaticText(self, wx.ID_ANY, str(Top_Stock[3]['lowPrice']))
        label_46.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_26.Add(label_46, 0, wx.ALIGN_CENTER, 0)
        sizer_23.Add(sizer_26, 1, wx.EXPAND, 0)
        label_47 = wx.StaticText(self, wx.ID_ANY, "CLOSE")
        label_47.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_27.Add(label_47, 0, wx.ALIGN_CENTER, 0)
        label_48 = wx.StaticText(self, wx.ID_ANY, str(Top_Stock[3]['ltp']))
        label_48.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_27.Add(label_48, 0, wx.ALIGN_CENTER, 0)
        sizer_23.Add(sizer_27, 1, wx.EXPAND, 0)
        label_49 = wx.StaticText(self, wx.ID_ANY, "VOLUME")
        label_49.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_28.Add(label_49, 0, wx.ALIGN_CENTER, 0)
        label_50 = wx.StaticText(self, wx.ID_ANY, str(Top_Stock[3]['tradedQuantity']))
        label_50.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_28.Add(label_50, 0, wx.ALIGN_CENTER, 0)
        sizer_23.Add(sizer_28, 1, wx.EXPAND, 0)
        label_51 = wx.StaticText(self, wx.ID_ANY, "TURNOVER")
        label_51.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_29.Add(label_51, 0, wx.ALIGN_CENTER, 0)
        label_52 = wx.StaticText(self, wx.ID_ANY, str(Top_Stock[3]['turnoverInLakhs']))
        label_52.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_29.Add(label_52, 0, wx.ALIGN_CENTER, 0)
        sizer_23.Add(sizer_29, 1, wx.EXPAND, 0)
        sizer_1.Add(sizer_23, 1, wx.EXPAND, 0)
        static_line_5 = wx.StaticLine(self, wx.ID_ANY)
        sizer_1.Add(static_line_5, 0, wx.ALL | wx.EXPAND, 5)
        label_53 = wx.StaticText(self, wx.ID_ANY, '\n'+Top_Stock[4]['symbol']+'\n')
        label_53.SetFont(wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 1, ""))
        sizer_1.Add(label_53, 0, 0, 0)
        label_54 = wx.StaticText(self, wx.ID_ANY, "OPEN")
        label_54.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_31.Add(label_54, 0, wx.ALIGN_CENTER, 0)
        label_55 = wx.StaticText(self, wx.ID_ANY, str(Top_Stock[4]['openPrice']))
        label_55.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_31.Add(label_55, 0, wx.ALIGN_CENTER, 0)
        sizer_30.Add(sizer_31, 1, wx.EXPAND, 0)
        label_56 = wx.StaticText(self, wx.ID_ANY, "HIGH")
        label_56.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_32.Add(label_56, 0, wx.ALIGN_CENTER, 0)
        label_57 = wx.StaticText(self, wx.ID_ANY, str(Top_Stock[4]['highPrice']))
        label_57.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_32.Add(label_57, 0, wx.ALIGN_CENTER, 0)
        sizer_30.Add(sizer_32, 1, wx.EXPAND, 0)
        label_58 = wx.StaticText(self, wx.ID_ANY, "LOW")
        label_58.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_33.Add(label_58, 0, wx.ALIGN_CENTER, 0)
        label_59 = wx.StaticText(self, wx.ID_ANY, str(Top_Stock[4]['lowPrice']))
        label_59.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_33.Add(label_59, 0, wx.ALIGN_CENTER, 0)
        sizer_30.Add(sizer_33, 1, wx.EXPAND, 0)
        label_60 = wx.StaticText(self, wx.ID_ANY, "CLOSE")
        label_60.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_34.Add(label_60, 0, wx.ALIGN_CENTER, 0)
        label_61 = wx.StaticText(self, wx.ID_ANY, str(Top_Stock[4]['ltp']))
        label_61.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_34.Add(label_61, 0, wx.ALIGN_CENTER, 0)
        sizer_30.Add(sizer_34, 1, wx.EXPAND, 0)
        label_62 = wx.StaticText(self, wx.ID_ANY, "VOLUME")
        label_62.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_35.Add(label_62, 0, wx.ALIGN_CENTER, 0)
        label_63 = wx.StaticText(self, wx.ID_ANY, str(Top_Stock[4]['tradedQuantity']))
        label_63.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_35.Add(label_63, 0, wx.ALIGN_CENTER, 0)
        sizer_30.Add(sizer_35, 1, wx.EXPAND, 0)
        label_64 = wx.StaticText(self, wx.ID_ANY, "TURNOVER")
        label_64.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_36.Add(label_64, 0, wx.ALIGN_CENTER, 0)
        label_65 = wx.StaticText(self, wx.ID_ANY, str(Top_Stock[4]['turnoverInLakhs']))
        label_65.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, 0, ""))
        sizer_36.Add(label_65, 0, wx.ALIGN_CENTER, 0)
        sizer_30.Add(sizer_36, 1, wx.EXPAND, 0)
        sizer_1.Add(sizer_30, 1, wx.EXPAND, 0)
        static_line_7 = wx.StaticLine(self, wx.ID_ANY)
        sizer_1.Add(static_line_7, 0, wx.ALL | wx.EXPAND, 5)
        self.SetSizer(sizer_1)
        self.Layout()
        # end wxGlade

# end of class MyFrame_Gainers

class MyApp_Gainers_Losers(wx.App):
    def OnInit(self):
        self.frame_Gainers_Losers = MyFrame_Gainers_Losers(None, wx.ID_ANY, "")
        self.SetTopWindow(self.frame_Gainers_Losers)
        self.frame_Gainers_Losers.Show()
        return True

# end of class MyApp_Gainers

Selection='None'
Top_Stock=[]

def Stock_Gainers_Losers(My_Selection):
    global Selection
    Selection=My_Selection
    nse = Nse()
    if Selection=='Gain':
        top = nse.get_top_gainers()
    else:
        top = nse.get_top_losers()
    global Top_Stock
    Top_Stock=top
    app_Gainers_Losers = MyApp_Gainers_Losers(0)
    app_Gainers_Losers.MainLoop()
