if(right_prediction[0]==0 and left_prediction[0]==0):
        computed_score=computed_score+1
else:
        computed_score=computed_score-1    
if(computed_score<0):
        computed_score=0   
cv2.putText(video_fra,'Score:'+str(computed_score),(100,height-20), text_font, 1,(255,255,255),1,cv2.LINE_AA)
if(computed_score>8):
        cv2.imwrite(os.path.join(path,'image.jpg'),video_fra)
        cv2.putText(video_fra, 'You are Drowsy', (80,height-60), text_font, 2, (209, 80, 0, 255), 3, cv2.LINE_AA)
        try:
            warning_alarm.play()
        except: 
            pass
        if(abc<16):
            abc= abc+2
        else:
            abc=abc-2
            if(abc<2):
                abc=2
        cv2.rectangle(video_fra,(0,0),(width,height),(0,0,255),abc) 
cv2.imshow('frame',video_fra)
if cv2.waitKey(1) & 0xFF == ord('q'):
        break
capture_video.release()
cv2.destroyAllWindows()
