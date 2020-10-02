#include "pch.h"
using namespace std;
typedef unsigned int bits32;

void missing_field_error()
{
	mexErrMsgTxt("conv_encoder_conf not correct!");
}

int get_int(mxArray* pm)
{
	if (mxIsDouble(pm))
		return (int)((mxGetDoubles(pm))[0]);
	else if (mxIsInt32(pm))
		return mxGetInt32s(pm)[0];
	else
		mexErrMsgTxt("Error! INT type expected.");
}

// input: double or logical 1*N matrix, N<=32.
bits32 convert_to_bits32(mxArray* pm)
{
	unsigned int N = mxGetN(pm);
	bits32 temp = 0;
	if (mxIsDouble(pm))
	{
		double* arr = mxGetDoubles(pm);
		for (int i = 0; i < N; i++)
		{
			temp <<= 1;
			if (arr[i] == 1.0)
				temp |= 0x1;
		}
	}
	else if (mxIsLogical(pm))
	{
		bool* arr = mxGetLogicals(pm);
		for (int i = 0; i < N; i++)
		{
			temp <<= 1;
			if (arr[i])
				temp |= 0x1;
		}
	}
	else
		mexErrMsgTxt("Error! binary elements expected.");
	return temp;
}

bits32 inner_prod_mod2(bits32 x, bits32 y)
{
	bits32 ans;
	ans = x & y;
	ans ^= (ans >> 16);
	ans ^= (ans >> 8);
	ans ^= (ans >> 4);
	ans ^= (ans >> 2);
	ans ^= (ans >> 1);
	return ans & 0x1;
}

int hamming_distance(bits32 x, bits32 y)
{
	return int(__popcnt64(x^y));
}

typedef struct
{
	bits32 next_state;
	bits32 curr_output;
}vST;

struct path_choice
{
	bits32 best_choice;
	bits32 last_state;
};

// T: loss type.
class vNet
{
public:
	vNet(int N_states, int length):N_states(N_states), length(length)
	{
		pc = new path_choice[N_states*length];
	}

	~vNet()
	{
		delete[] pc;
	}

	path_choice& operator()(int pos, int state_index)
	{
		return *(pc + state_index + pos * N_states);
	}

	int N_states;
	int length;
	path_choice* pc;
};

class ViterbiEncoderDecoder
{
public:
	ViterbiEncoderDecoder(int n, int k, int N, bits32* A_arr) :n(n), k(k), N(N), A_arr(A_arr)
	{
		N_inner_state_bits = (N - 1)*k;
		N_related_bits = N * k;
		N_inner_states = (0x1 << N_inner_state_bits);
		N_choices = (0x1 << k);

		// Construct State Transition Table(STT).
		STT = new vST[N_inner_states*N_choices];
		for (bits32 si = 0; si < N_inner_states; si++)
		{
			for (bits32 ci = 0; ci < N_choices; ci++)
			{
				vST* t = STT_indexing(si, ci);
				bits32 curr_full_state = si | (ci << N_inner_state_bits);

				t->next_state = (si >> k) | (ci << (N_inner_state_bits - k));
				t->curr_output = 0x0;
				for (int iter = 0; iter < n; iter++)
				{
					(t->curr_output) <<= 1;
					(t->curr_output) |= inner_prod_mod2(curr_full_state, A_arr[iter]);
				}
			}
		}

		pvNet = nullptr;
	}

	bool is_same_params(int _n, int _k, int _N, bits32* _A_arr)
	{
		if (_n != n || _k != k || _N != N) return false;
		// Check for _A_arr;
		for (int i = 0; i < n; i++)
		{
			if (A_arr[i] != _A_arr[i])return false;
		}
		return true;
	}

	~ViterbiEncoderDecoder()
	{
		delete pvNet;
		delete STT;
		delete A_arr;
	}

	// Perform Viterbi Decode.
	bool* HardDecode(bool* input_bits, int input_length, bool trailing, int& N_output_bits)
	{
		if ((input_length % n))
			mexErrMsgTxt("Decoder: Input length not multiples of n.");
		int input_blocks = input_length / n;

		// Step0: Construct Viterbi memory space (vNet) if necessary.
		if (!pvNet || (pvNet->length != input_blocks) || (pvNet->N_states != N_inner_states))
		{
			delete pvNet;	// It's OK to delete a nullptr.
			pvNet = new vNet(N_inner_states, input_blocks);
		}

		N_output_bits = input_blocks * k;
		int* losses = new int[N_inner_states];
		int* losses_new = new int[N_inner_states];
		for (int i = 0; i < N_inner_states; i++)losses[i] = N_output_bits;
		losses[0] = 0;	// initial state is 000.

		for (int block = 0; block < input_blocks; block++)
		{
			for (int i = 0; i < N_inner_states; i++)losses_new[i] = -1;	// Set new losses to "undefined".
			// Step1: Fetch this block and convert to bits32 format.
			bits32 curr_block = 0;
			int input_index = block * n;
			for (int n_iter = 0; n_iter < n; n_iter++)
			{
				curr_block <<= 1;
				if (input_bits[input_index + n_iter]) curr_block |= 0x1;
			}

			// Step2: One viterbi step on vNet.
			for(bits32 si=0;si<N_inner_states;si++)
				for (bits32 ci = 0; ci < N_choices; ci++)
				{
					vST* t = STT_indexing(si, ci);
					int loss_incr = hamming_distance(t->curr_output, curr_block);
					if (losses_new[t->next_state] == -1)
					{
						losses_new[t->next_state] = loss_incr + losses[si];
						(*pvNet)(block, t->next_state) = { ci, si };	// choice, last_state
					}
					else
					{
						if (losses_new[t->next_state] > loss_incr + losses[si])
						{
							losses_new[t->next_state] = loss_incr + losses[si];
							(*pvNet)(block, t->next_state) = { ci, si };
						}
					}
				}

			// Step3: Update losses.
			swap(losses_new, losses);
		}

		// Step4: Perform HARD decoding.
		
		bool* ret_bits = new bool[N_output_bits];

		bits32 p = 0;
		if (!trailing)
		{
			// Find the index of minimum in "losses".
			for (int i = 1; i < N_inner_states; i++)
				if (losses[p] > losses[i])p = i;
		}

		for (int block = input_blocks - 1; block >= 0; --block)
		{
			int output_index = block * k;
			path_choice& pc = (*pvNet)(block, p);
			p = pc.last_state;
			bits32 partial_decode = pc.best_choice;
			for (int i = 0; i < k; i++)
			{
				ret_bits[output_index + i] = partial_decode & 0x1;
				partial_decode >>= 1;
			}
		}

		delete[] losses;
		delete[] losses_new;
		return ret_bits;
	}

	bool* SoftDecode(double* input_syms_losses, int input_blocks, bool trailing, int& N_output_bits)
	{
		// Step0: Construct Viterbi memory space (vNet) if necessary.
		if (!pvNet || (pvNet->length != input_blocks) || (pvNet->N_states != N_inner_states))
		{
			delete pvNet;	// It's OK to delete a nullptr.
			pvNet = new vNet(N_inner_states, input_blocks);
		}

		N_output_bits = input_blocks * k;
		double* losses = new double[N_inner_states];
		double* losses_new = new double[N_inner_states];
		for (int i = 0; i < N_inner_states; i++)losses[i] = double(N_output_bits);
		losses[0] = 0.0;		// initial state is 000.

		int N_output_possibilities = (0x1 << n);	
		for (int block = 0; block < input_blocks; block++)
		{
			for (int i = 0; i < N_inner_states; i++)losses_new[i] = -1;	// Set new losses to "undefined".
			// Step1: Fetch this block.
			int input_index = block * n;
			double* curr_block_probs = input_syms_losses + block * N_output_possibilities;

			// Step2: One viterbi step on vNet.
			for (bits32 si = 0; si < N_inner_states; si++)
				for (bits32 ci = 0; ci < N_choices; ci++)
				{
					vST* t = STT_indexing(si, ci);
					// double loss_incr = hamming_distance(t->curr_output, curr_block);
					double loss_incr = curr_block_probs[t->curr_output];
					if (loss_incr < 0)mexErrMsgTxt("Soft decode error: loss must be positive.");

					if (losses_new[t->next_state] == -1)
					{
						losses_new[t->next_state] = loss_incr + losses[si];
						(*pvNet)(block, t->next_state) = { ci, si };	// choice, last_state
					}
					else
					{
						if (losses_new[t->next_state] > loss_incr + losses[si])
						{
							losses_new[t->next_state] = loss_incr + losses[si];
							(*pvNet)(block, t->next_state) = { ci, si };
						}
					}
				}

			// Step3: Update losses.
			swap(losses_new, losses);
		}

		// Step4: Perform HARD decoding.

		bool* ret_bits = new bool[N_output_bits];

		bits32 p = 0;
		if (!trailing)
		{
			// Find the index of minimum in "losses".
			for (int i = 1; i < N_inner_states; i++)
				if (losses[p] > losses[i])p = i;
		}

		for (int block = input_blocks - 1; block >= 0; --block)
		{
			int output_index = block * k;
			path_choice& pc = (*pvNet)(block, p);
			p = pc.last_state;
			bits32 partial_decode = pc.best_choice;
			for (int i = 0; i < k; i++)
			{
				ret_bits[output_index + i] = partial_decode & 0x1;
				partial_decode >>= 1;
			}
		}

		delete[] losses;
		delete[] losses_new;
		return ret_bits;
	}

private:
	vST* STT_indexing(int state_index, int choice_index)
	{
		return STT + state_index * N_choices + choice_index;
	}

	int n, k, N;
	bits32* A_arr;
	int N_related_bits;
	int N_inner_state_bits;
	int N_inner_states;
	int N_choices;
	vST* STT;

	vNet* pvNet;
};

ViterbiEncoderDecoder* ved = nullptr;

// Entry Function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	bool b_soft_decode = false;
	if (nlhs != 1)
		mexErrMsgTxt("Only 1 output vector expected.");
	if (nrhs != 3 && nrhs != 2)
	{
		mexErrMsgTxt("2 or 3 input parameters expected.");
	}
	const mxArray* encoded_bits = prhs[0];				// type: logical
	const mxArray* conv_encoder_conf = prhs[1];	// type: struct


	// Parse prhs[0]: logical array.
	size_t encoded_bits_len;
	bool* logicals = nullptr;
	double* input_syms_probs = nullptr;	// initialize as null.
	if (mxIsLogical(encoded_bits))
	{
		size_t t = mxGetM(encoded_bits);
		if (t != 1)
			mexErrMsgTxt("Encoded bits must be a row vector.");
		encoded_bits_len = mxGetN(encoded_bits);
		logicals = mxGetLogicals(encoded_bits);
	}
	else if (mxIsDouble(encoded_bits))
	{
		encoded_bits_len = mxGetN(encoded_bits);
		input_syms_probs = mxGetDoubles(encoded_bits);
	}
	else
		mexErrMsgTxt("encoded_bits must be an logical array or a double matrix.");


	// Parse prhs[1]: struct
	if (!mxIsStruct(conv_encoder_conf))
		mexErrMsgTxt("conv_encoder_conf must be a struct.");
	mxArray* n, *k, *N, *A, *trailing;
	n = mxGetField(conv_encoder_conf, 0, "n"); if (!n)missing_field_error();
	k = mxGetField(conv_encoder_conf, 0, "k"); if (!k)missing_field_error();
	N = mxGetField(conv_encoder_conf, 0, "N"); if (!N)missing_field_error();
	A = mxGetField(conv_encoder_conf, 0, "A"); if (!A)missing_field_error();
	trailing = mxGetField(conv_encoder_conf, 0, "trailing"); if (!trailing)missing_field_error();

	int i_n, i_k, i_N; // only double and int32 supported.
	i_n = get_int(n);
	i_k = get_int(k);
	i_N = get_int(N);

	// Get A matrix and convert into bits32 type.
	if (!mxIsCell(A))
		mexErrMsgTxt("conv_encoder_conf.A must be a cell.");
	if (mxGetM(A) != i_n || mxGetN(A) != 1)
		mexErrMsgTxt("conv_encoder_conf.A must be in size cell{n, 1}.");

	int N_related_bits = i_N * i_k;
	if (N_related_bits > 32)
		mexErrMsgTxt("Related bits=N*k>32 is not supported.");

	bits32* b32_A = new bits32[i_n];				// Notice: New array here!
	for (int i_n_iter = 0; i_n_iter < i_n; i_n_iter++)
	{
		b32_A[i_n_iter] = convert_to_bits32(mxGetCell(A, i_n_iter));
	}

	if (!mxIsLogical(trailing))mexErrMsgTxt("conv_encoder_conf.trailing must be Logical.");
	bool b_trailing = *(mxGetLogicals(trailing));

	// Parse prhs[2] if it exists.
	if (3 == nrhs)
	{
		const mxArray* soft_decode = prhs[2];				// type: logical/bool
		if (!mxIsLogical(soft_decode)) mexErrMsgTxt("soft_decode must be of Logical type.");
		b_soft_decode = *(mxGetLogicals(soft_decode));
	}

	/* Initialize viterbi encoder-decoder and calculate State-Transition Table.. */
	if (!ved || !(ved->is_same_params(i_n, i_k, i_N, b32_A)))
	{
		// if ved does not exists or the params are not the same, then it should be re-constructed.
		delete ved;
		ved = new ViterbiEncoderDecoder(i_n, i_k, i_N, b32_A);
		// pointer b32_A is managed by ViterbiEncoderDecoder object.
	}
	else
		delete[] b32_A;

	int output_bits = 0;
	size_t two_int_arr[2];
	two_int_arr[0] = 1;
	bool* ret_bits = nullptr;

	if (!b_soft_decode)
		ret_bits = ved->HardDecode(logicals, encoded_bits_len, b_trailing, output_bits);
	else
		// HERE: encoded_bits_len means encoded_syms_len.
		ret_bits = ved->SoftDecode(input_syms_probs, encoded_bits_len, b_trailing, output_bits);

	if (b_trailing)
		(output_bits -= (N_related_bits)) += i_k;

	two_int_arr[1] = output_bits;
	plhs[0] = mxCreateLogicalArray(2, two_int_arr);
	bool* ptr = mxGetLogicals(plhs[0]);
	bool* p = ret_bits;
	for (int i = 0; i < output_bits; i++)	// copy the bools.
		*(ptr++) = *(p++);
	delete[] ret_bits;

	return;
}

// DLL System Entry. May do some cleanup work here.
BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserver)
{

	switch (ul_reason_for_call)
	{
	case DLL_PROCESS_ATTACH:
		//GetModuleHandle: Process handle got. Instance handle got.
		break;
	case DLL_THREAD_ATTACH:
		break;
	case DLL_THREAD_DETACH:
		break;
	case DLL_PROCESS_DETACH:
		break;
	}
	return TRUE;
}