import sys
import typer
import orjson
import ocr_map as om

app = typer.Typer()

Verbose = typer.Option(False, '-v', '--verbose')
Model = typer.Option(..., '-m', '--model', help='Path to pickled model')
TopPreds = typer.Option(25, '-k', '--top-preds', help='Number of top predictions to output')

@app.callback()
def callback():
  ...

@app.command()
def fit(pickle_output: str, verbose: bool = Verbose, k: int = TopPreds):
  """Fits a model to samples read from stdin and saves it to a pickle file.
  Expected input format: `{"label": "...", "preds": [["pred1", 0.91], ["pred2", 0.01], ...]}`.
  """
  if verbose:
    print('Parsing samples from stdin...')

  def samples():
    i = 0
    for i, line in enumerate(sys.stdin):
      if verbose and i % 1000 == 0:
        print(f'\rParsing... {i}', end='', flush=True)
      obj = orjson.loads(line)
      yield om.Sample(obj['label'], obj['preds'][:k])
    if verbose:
      print(f'\rParsing... {i}', end='', flush=True)
  
  model = om.Model.fit(samples())
  if verbose:
    print(f'Saving model to {pickle_output}...')
  import pickle
  with open(pickle_output, 'wb') as f:
    pickle.dump(model, f)


@app.command()
def denoise(model_path: str = Model, verbose: bool = Verbose, k: int = TopPreds):
  """Denoises samples read from stdin and writes them to stdout. Both input and outputs are a list of distributions per line"""
  if verbose:
    print(f'Loading model from "{model_path}"...', file=sys.stderr)
  model = om.Model.unpickle(model_path)
  from functools import cache
  model.posterior = cache(model.posterior)

  for i, line in enumerate(sys.stdin):
    outputs = []
    for j, distrib in enumerate(orjson.loads(line)):
      if verbose:
        print(f'\rDenoising... {i}:{j}', end='', flush=True, file=sys.stderr)
      outputs.append(model.denoise(distrib).most_common(k))
      
    sys.stdout.buffer.write(orjson.dumps(outputs) + b'\n')


@app.command()
def simulate(model_path: str = Model, verbose: bool = Verbose, k: int = TopPreds):
  """Simulates denoised OCR predictions given each label. Expects space-delimited labels on every line. Outputs a list of distributions per line."""
  if verbose:
    print(f'Loading model from "{model_path}"...', file=sys.stderr)
  model = om.Model.unpickle(model_path)

  for i, line in enumerate(sys.stdin):
    labels = line.strip().split()
    outputs = []
    for j, label in enumerate(labels):
      if verbose:
        print(f'\rSimulating... {i}:{j}', end='', flush=True, file=sys.stderr)
      outputs.append(model.simulate(label).most_common(k))
      
    sys.stdout.buffer.write(orjson.dumps(outputs) + b'\n')