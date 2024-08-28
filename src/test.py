import pickle

from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from tqdm import tqdm

from chains import chains


def compute_bleu(reference, hypothesis):
    reference = [reference.split()]
    hypothesis = hypothesis.split()
    return sentence_bleu(reference, hypothesis)


def compute_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores


def get_test_dataset(file_path: str) -> list[tuple[str, str]]:
    with open(file_path, "r") as file:
        text = file.read()

    dataset = []
    pairs = text.split("-----")
    pairs = [pairs.split("---") for pairs in pairs]

    for inp, out in pairs:
        inp = "\n".join(e.strip() for e in inp.split("\n") if not e.startswith("//"))
        out = "\n".join(e.strip() for e in out.split("\n") if not e.startswith("//"))
        dataset.append((inp.strip(), out.strip()))

    return dataset


def generate_outputs(chains, dataset: list[tuple[str, str]]) -> dict[str, list[str]]:
    llm_outputs = {chain.name: [] for chain in chains}  # dict[str, list[str]]

    for chain in chains:
        for input_text, _ in tqdm(dataset, desc=f"Generating outputs for {chain.name}"):
            output = chain.invoke(input_text)
            llm_outputs[chain.name].append(output)

        assert len(llm_outputs[chain.name]) == len(
            dataset
        ), f"The number of generated outputs does not match the number of inputs for {chain.name}. Number of inputs: {len(dataset)}, number of outputs: {len(llm_outputs[chain.name])}"

    return llm_outputs


if __name__ == "__main__":
    for scenario in range(1, 5):
        print(f"Running scenario {scenario}")
        file_path = f"./test_data/q{scenario}.txt"
        dataset = get_test_dataset(file_path)  # list[tuple[str, str]]

        llm_outputs = generate_outputs(chains, dataset)

        # save the generated outputs
        with open(f"generated_outputs/llm_outputs_q{scenario}.pkl", "wb") as f:
            pickle.dump(llm_outputs, f)
        for chain in chains:
            with open(f"generated_outputs/{chain.name}_q{scenario}.txt", "w") as f:
                f.write("\n\n".join(llm_outputs[chain.name]))  # type: ignore

    for scenario in range(1, 5):
        print(f"Running scenario {scenario}")
        with open(f"generated_outputs/llm_outputs_q{scenario}.pkl", "rb") as f:
            llm_outputs = pickle.load(f)
        dataset = get_test_dataset(f"./test_data/q{scenario}.txt")

        bleu_scores = {llm: [] for llm in llm_outputs}
        rouge_scores = {llm: {"rouge1": [], "rougeL": []} for llm in llm_outputs}

        for input_text, desired_output in dataset:
            for llm, outputs in llm_outputs.items():
                # Assume the corresponding output is in the same order as the dataset
                output = outputs[dataset.index((input_text, desired_output))]
                bleu = compute_bleu(desired_output, output)
                rouge = compute_rouge(desired_output, output)
                bleu_scores[llm].append(bleu)
                rouge_scores[llm]["rouge1"].append(rouge["rouge1"].fmeasure)
                rouge_scores[llm]["rougeL"].append(rouge["rougeL"].fmeasure)

        # Calculate average scores for each LLM
        avg_bleu_scores = {
            llm: sum(scores) / len(scores) for llm, scores in bleu_scores.items()
        }
        avg_rouge_scores = {
            llm: {
                metric: sum(scores) / len(scores) for metric, scores in metrics.items()
            }
            for llm, metrics in rouge_scores.items()
        }

        # Print results
        print("Average BLEU Scores:")
        for llm, score in avg_bleu_scores.items():
            print(f"{llm}: {score}")

        print("\nAverage ROUGE Scores:")
        for llm, scores in avg_rouge_scores.items():
            print(f"{llm}:")
            for metric, score in scores.items():
                print(f"  {metric}: {score}")
