from subprocess import call

call(['wget', 'https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat'])
call(['mkdir', '-vp', 'models'])
call(['mv', '-v', 'shape_predictor_68_face_landmarks.dat', 'models'])
