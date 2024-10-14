#reverses normalization process (for GAN outputs)
def denormalize_output(x_music_out, x_durations_out, x_music_in, x_durations_in):

  a = (x_music_out + 1) / 2
  x_music_updated_out = (a * np.amax(x_music_in)) - (a *np.amin(x_music_in)) +  np.amin(x_music_in)

  b = (x_durations_out + 1) / 2
  x_durations_updated_out = (b * np.amax(x_durations_in)) - (b *np.amin(x_durations_in)) +  np.amin(x_durations_in)

  x_music_updated_out = (np.rint(x_music_updated_out)).astype(int)
  x_durations_updated_out = (np.rint(x_durations_updated_out)).astype(int)

  return x_music_updated_out, x_durations_updated_out

#unmixes the output into two (16,1) arrays
def musicdurationsunmix(music):
  musiclist = []
  durationslist = []
  for x in music.tolist():
    #musiclist.append([x[int(sequence_length/2-1)]])
    musiclist.append(x[1])
    #durationslist.append([x[int(sequence_length/2)]])
    durationslist.append(x[2])

  music_out = np.array(musiclist)
  durations_out = np.array(durationslist)

  return music_out, durations_out

#full deprocessing
def deprocess(music, x_music_in, x_durations_in):
  reshaped_music = music.reshape(16,3)
  music_out, durations_out = musicdurationsunmix(reshaped_music)
  x_music_updated_out, x_durations_updated_out = denormalize_output(music_out, durations_out, x_music_in, x_durations_in)
  return x_music_updated_out, x_durations_updated_out
