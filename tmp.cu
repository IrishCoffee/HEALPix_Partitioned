void singleCM(PIX_NODE h_ref_node[], int ref_N, PIX_NODE h_sam_node[], int sam_N, int h_sam_match[],int h_sam_matchedCnt[])
{
	//the maximum number of sample points that can be matched each time by each card
	int part_sam_N = 25000000;
	int part_ref_N = 8 * part_sam_N;

	PIX_NODE *d_ref_node[GPU_N];
	PIX_NODE *d_sam_node[GPU_N];
	int *d_sam_match[GPU_N], *d_sam_matchedCnt[GPU_N];

	int chunk_N = (int)ceil(sam_N * 1.0 / part_sam_N);
	int chunk_id = 0;

	omp_set_num_threads(GPU_N);
#pragma omp parallel
	{
		int i = omp_get_thread_num() % GPU_N;
		checkCudaErrors(cudaSetDevice(i));
		checkCudaErrors(cudaDeviceReset());

		size_t free_mem,total_mem;
		checkCudaErrors(cudaMemGetInfo(&free_mem,&total_mem));
		printf("Card %d before malloc %.2lf GB, total memory %.2lf GB\n",i,free_mem * 1.0 / GBSize,total_mem * 1.0 / GBSize);


		checkCudaErrors(cudaMalloc(&d_ref_node[i],sizeof(PIX_NODE) * part_ref_N));
		checkCudaErrors(cudaMalloc(&d_sam_node[i],sizeof(PIX_NODE) * part_sam_N));
		checkCudaErrors(cudaMalloc(&d_sam_match[i],sizeof(int) * part_sam_N  * 5));
		checkCudaErrors(cudaMalloc(&d_sam_matchedCnt[i],sizeof(int) * part_sam_N));

		checkCudaErrors(cudaMemGetInfo(&free_mem,&total_mem));
		printf("Card %d after malloc %.2lf GB, total memory %.2lf GB\n",i,free_mem * 1.0 / GBSize,total_mem * 1.0 / GBSize);

		while(chunk_id < chunk_N)
		//the total number of sample points processed by this card
		{
#pragma omp atomic
			chunk_id++;

			int cur_sam_N;
			if(chunk_id == chunk_N) // the last round
				cur_sam_N = sam_N - (chunk_id - 1) * part_sam_N;
			else
				cur_sam_N = part_sam_N;

			int start_sam_pos = (chunk_id - 1) * part_sam_N;
			int end_sam_pos = start_sam_pos + cur_sam_N - 1;

			int start_pix = h_sam_node[start_sam_pos].pix;
			int end_pix = h_sam_node[end_sam_pos].pix;

			int start_ref_pos;
			if(start_pix == 0)
				start_ref_pos = 0;
			else
				start_ref_pos = binary_search(start_pix - 1,h_ref_node,ref_N);
			//				start_ref_pos = get_start(start_pix,h_ref_node,ref_N);

			if(start_ref_pos == -1)
				continue;
			int end_ref_pos = binary_search(end_pix,h_ref_node,ref_N) - 1;
			if(end_ref_pos == -2)
				end_ref_pos = ref_N - 1;
			int cur_ref_N = end_ref_pos - start_ref_pos + 1;

			dim3 block(block_size);
			dim3 grid(min(65536,(int)ceil(cur_sam_N * 1.0 / block.x)));

			if(cur_ref_N == 0)
				continue;

			printf("\n\nCard %d chunk-%d\n",i,chunk_id - 1);
			printf("block.x %d grid.x %d\n",block.x,grid.x);
			printf("start_sam_pos %d start_sam_pix %d end_sam_pos %d end_sam_pix %d sam_N %d\n",start_sam_pos,start_pix,end_sam_pos,end_pix,cur_sam_N);
			printf("start_ref_pos %d start_ref_pix %d end_ref_pos %d end_ref_pix %d ref_N %d\n",start_ref_pos,h_ref_node[start_ref_pos].pix,end_ref_pos,h_ref_node[end_ref_pos].pix,cur_ref_N);
			checkCudaErrors(cudaMemset(d_sam_matchedCnt[i],0,sizeof(int) * part_sam_N));
			checkCudaErrors(cudaMemcpy(d_sam_node[i],h_sam_node + start_sam_pos,cur_sam_N * sizeof(PIX_NODE),cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(d_ref_node[i],h_ref_node + start_ref_pos,cur_ref_N * sizeof(PIX_NODE), cudaMemcpyHostToDevice));
			kernel_singleCM<<<grid,block>>>(d_ref_node[i],cur_ref_N,d_sam_node[i],cur_sam_N,d_sam_match[i],d_sam_matchedCnt[i],start_ref_pos,start_sam_pos);
			checkCudaErrors(cudaMemcpy(h_sam_matchedCnt + start_sam_pos,d_sam_matchedCnt[i],cur_sam_N * sizeof(int),cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaMemcpy(h_sam_match + start_sam_pos * 5,d_sam_match[i],cur_sam_N * 5 * sizeof(int),cudaMemcpyDeviceToHost));
		}
	}
	unsigned long long sum = 0;
	int cnt[1000];
	memset(cnt,0,sizeof(cnt));
	for(int i = sam_N - 1; i >= 0; --i)
	{
		sum += h_sam_matchedCnt[i];
		/*
		   cout << i << " " << h_sam_matchedCnt[i] << endl;
		   cout << h_sam_node[i].ra << " " << h_sam_node[i].dec << endl;
		   cout << "\n----------------\n" << endl;
		   for(int j = i * 5; j < i * 5 + min(5,h_sam_matchedCnt[i]); ++j)
		   {
		   int pos = h_sam_match[j];
		   cout << h_ref_node[pos].ra << " " << h_ref_node[pos].dec << endl;
		   }
		   cout << "\n--------------------\n" << endl;
		 */
	}
	cout << "sum " << sum << endl;
	cout << "ave " << sum * 1.0 / sam_N << endl;
}

