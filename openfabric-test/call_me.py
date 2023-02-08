from training_data import response
 
texts = ['hi', 'who made gravity', 'what is the size of earth']
output = []
for text in texts:
        # TODO Add code here
        res = response(text)
        output.append(res)

print(output)
        
