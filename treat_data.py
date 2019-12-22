import sys

file_path = 'log/2.txt'
final_file_path = 'log/data/2.txt'
if len(sys.argv) > 1:
    file_path = 'log/' + sys.argv[1] + '.txt'
    final_file_path = 'log/data/' + sys.argv[1] + '.txt'

train_epoch_list = []
train_loss_list = []
test_epoch_list = []
test_loss_list = []

i = 0

with open(file_path, "r") as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith('train'):
            epoch, loss = line[6:].strip().split(',')
            train_epoch_list.append(epoch)
            train_loss_list.append(loss)
        elif line.startswith('test'):
            epoch, loss = line[5:].strip().split(',')
            test_epoch_list.append(epoch)
            test_loss_list.append(loss)
        
        i += 1
        if (i % 1000 == 0):
            print(i)

print('-----------------------------------------')
print('total line num: %d' % i)
print('train data len: %d' % len(train_loss_list))
print('test data len: %d' % len(test_loss_list))

f = open(final_file_path, mode='w', encoding='utf-8')  
f.write(','.join(train_epoch_list))
f.write('\n')
f.write(','.join(train_loss_list))
f.write('\n')  
f.write(','.join(test_epoch_list))
f.write('\n')  
f.write(','.join(test_loss_list))
f.write('\n')            
f.close()                                     