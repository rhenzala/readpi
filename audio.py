import alsaaudio

def volume_up():
    mixer = alsaaudio.Mixer()
    current_volume = mixer.getvolume()[0]
    new_volume = max(0, min(100, current_volume + 10))
    mixer.setvolume(new_volume)

def volume_down():
    mixer = alsaaudio.Mixer()
    current_volume = mixer.getvolume()[0]
    new_volume = max(0, min(100, current_volume - 10))
    mixer.setvolume(new_volume)

def set_volume(volume_level):
    mixer = alsaaudio.Mixer()
    volume_level = max(0, min(100, volume_level))
    mixer.setvolume(volume_level)

