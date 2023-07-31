
#### pyqt 설치 ####
conda install pyqt
-----------------------------------------------------------------------------

#### deginer install ####
sudo apt install qttools5-dev-tools
-----------------------------------------------------------------------------




#### ui파일-> py파일 ####
pyuic5 -x (ui파일 위치및 파일이름) -o (바꾼 파일 정할 위치및 파일이름과 바꿀 파일 유형) 

ex) pyuic5 -x /home/pi/code/untitled.ui -o /home/pi/code/untitled.py

-----------------------------------------------------------------------------

deginer로 만든 ui파일을 py파일로 바꾸고 vs code에서 ipynb파일형식으로 파일을 하나 만든 뒤에 py파일에 있는 내용을 복사해서 붙여넣는다 (커널 pyqt가 설치된 커널에서 작동된다)



-------------------------------------------------------------------------------

self.label.setText("Test set Text") #텍스트 변환
self.label.setFont(QtGui.QFont("궁서",20)) #폰트,크기 조절
self.label.setStyleSheet("Color : green") #글자색 변환