package main

import (
	"fmt"
	"math"
	"slices"
	"sort"
	"strconv"
	"strings"
)

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func longestMonotonicSubarray(nums []int) int {
	if len(nums) == 1 {
		return 1
	}
	var maxLength int = 1
	var upLength int = 1
	var downLength int = 1
	for i := 1; i < len(nums); i++ {
		if nums[i] > nums[i-1] {
			upLength++
			downLength = 1
		}
		if nums[i] < nums[i-1] {
			downLength++
			upLength = 1
		}
		if nums[i] == nums[i-1] {
			downLength = 1
			upLength = 1
		}
		if downLength > maxLength {
			maxLength = downLength
		}
		if upLength > maxLength {
			maxLength = upLength
		}
	}
	return maxLength
}

func singleNumber(nums []int) int {
	if len(nums) == 1 {
		return nums[0]
	}
	var xorElement = nums[0]
	for i := 1; i < len(nums); i++ {
		xorElement = xorElement ^ nums[i]
	}
	return xorElement
}

func maxSubarraySumCircular(nums []int) int {
	var maxSum = nums[0]
	var minSum = nums[0]
	var tempMax = 0
	var tempMin = 0
	var totalSum = 0

	for i := 0; i < len(nums); i++ {
		tempMax = max(tempMax, 0) + nums[i]
		maxSum = max(tempMax, maxSum)

		tempMin = min(tempMin, 0) + nums[i]
		minSum = min(tempMin, minSum)

		totalSum += nums[i]
	}

	if minSum == totalSum {
		return maxSum
	} else {
		return max(maxSum, totalSum-minSum)
	}
}

func candy(ratings []int) int {
	var n = len(ratings)
	var totalCandies = n
	for i := 1; i < n; {
		if ratings[i] == ratings[i-1] {
			i++
			continue
		}
		var currentPeek = 0
		for i < n && ratings[i] > ratings[i-1] {
			currentPeek++
			totalCandies += currentPeek
			i++
		}
		if i == n {
			return totalCandies
		}
		var currentValley = 0
		for i < n && ratings[i] < ratings[i-1] {
			currentValley++
			totalCandies += currentValley
			i++
		}
		totalCandies -= min(currentValley, currentPeek)
	}
	return totalCandies
}

func coinChange(coins []int, amount int) int {
	var dp = make([]int, amount+1)
	for i := 0; i < len(dp); i++ {
		dp[i] = amount + 1
	}
	dp[0] = 0
	for i := 1; i < len(dp); i++ {
		for j := 0; j < len(coins); j++ {
			if i-coins[j] >= 0 {
				dp[i] = min(dp[i], dp[i-coins[j]]+1)
			}
		}
	}
	return dp[amount]
}

var happenedNumber []int

func getTotalSquare(number int) int {
	var res int = 0
	if number == 1 {
		return 1
	}
	for number > 0 {
		res += (number % 10) * (number % 10)
		number /= 10
	}
	if slices.Contains(happenedNumber, res) {
		return -1
	}
	happenedNumber = append(happenedNumber, res)
	return getTotalSquare(res)
}

func isHappy(n int) bool {
	return getTotalSquare(n) == 1
}

func maxAscendingSum(nums []int) int {
	var maxSum int = nums[0]
	var tempSum int = nums[0]
	for i := 1; i < len(nums); i++ {
		if nums[i] > nums[i-1] {
			tempSum += nums[i]
		} else {
			if tempSum > maxSum {
				maxSum = tempSum
			}
			tempSum = nums[i]
		}
		if tempSum > maxSum {
			maxSum = tempSum
		}
	}
	return maxSum
}
func areAlmostEqual(s1 string, s2 string) bool {
	var diffCount = 0
	var diffColls1 = make([]byte, 0)
	var diffColls2 = make([]byte, 0)
	for i := 0; i < len(s1); i++ {
		if s1[i] != s2[i] {
			diffCount++
			diffColls2 = append(diffColls2, s2[i])
			diffColls1 = append(diffColls1, s1[i])
		}
		if diffCount > 2 {
			return false
		}
	}
	if diffCount == 2 {
		return diffColls1[0] == diffColls2[1] && diffColls1[1] == diffColls2[0]
	}
	if diffCount != 0 {
		return false
	} else {
		return true
	}
}

func tupleSameProduct(nums []int) int {
	var freqMap = make(map[int]int)
	for i := 0; i < len(nums)-1; i++ {
		for j := i + 1; j < len(nums); j++ {
			if freqMap[nums[i]*nums[j]] == 0 {
				freqMap[nums[i]*nums[j]] = 1
			} else {
				freqMap[nums[i]*nums[j]]++
			}
		}
	}
	var count int = 0
	for _, value := range freqMap {
		if value > 1 {
			count += 4 * value * (value - 1)
		}
	}
	return count
}

func queryResults(limit int, queries [][]int) []int {
	var result = make([]int, len(queries))
	var freqMap = make(map[int]int)
	var coloredBall = make(map[int]int)
	var diffColorCount = 0
	for i := 0; i < len(queries); i++ {
		if coloredBall[queries[i][0]] == 0 {
			if freqMap[queries[i][1]] == 0 {
				diffColorCount++
			}
		} else {
			var oldColor = coloredBall[queries[i][0]]
			var newColor = queries[i][1]
			freqMap[oldColor]--
			if freqMap[oldColor] == 0 {
				diffColorCount--
			}
			if freqMap[newColor] == 0 {
				diffColorCount++
			}
		}
		coloredBall[queries[i][0]] = queries[i][1]
		freqMap[queries[i][1]]++
		result[i] = diffColorCount
	}
	return result
}

type NumberContainers struct {
	numbers map[int]int
	index   map[int][]int
}

func NumberContainersConstructor() NumberContainers {
	return NumberContainers{
		numbers: make(map[int]int),
		index:   make(map[int][]int),
	}
}

func (this *NumberContainers) Change(index int, number int) {
	var oldValue = this.numbers[index]
	this.numbers[index] = number
	if oldValue != 0 {
		this.index[oldValue] = DeleteSorted(this.index[oldValue], index)
	}
	this.index[number] = InsertSorted(this.index[number], index)
}

func (this *NumberContainers) Find(number int) int {
	if this.index[number] != nil {
		return -1
	} else {
		return this.index[number][0]
	}
}

func InsertSorted(arr []int, val int) []int {
	idx := sort.SearchInts(arr, val)
	arr = append(arr, 0)
	copy(arr[idx+1:], arr[idx:])
	arr[idx] = val
	return arr
}

func DeleteSorted(arr []int, val int) []int {
	idx := sort.SearchInts(arr, val)
	if idx < len(arr) && arr[idx] == val {
		arr = append(arr[:idx], arr[idx+1:]...)
	}
	return arr
}

func countBadPairs(nums []int) int64 {
	var notBadPairCount = 0
	var mapper = make(map[int]int)
	for i := 0; i < len(nums); i++ {
		mapper[nums[i]-i]++
		notBadPairCount += mapper[nums[i]-i]
	}

	return int64(len(nums)*(len(nums)-1)/2 - notBadPairCount)
}

type stack[T any] struct {
	Push   func(T)
	Pop    func() T
	Length func() int
}

func Stack[T any]() stack[T] {
	slice := make([]T, 0)
	return stack[T]{
		Push: func(i T) {
			slice = append(slice, i)
		},
		Pop: func() T {
			res := slice[len(slice)-1]
			slice = slice[:len(slice)-1]
			return res
		},
		Length: func() int {
			return len(slice)
		},
	}
}
func solve(index int, digits string, comb []string, ans *[]string, temp string) {
	if index == len(digits) {
		*ans = append(*ans, temp)
		return
	}

	for _, ch := range comb[digits[index]-'0'] {
		solve(index+1, digits, comb, ans, temp+string(ch))
	}
}

func letterCombinations(digits string) []string {
	if len(digits) == 0 {
		return []string{}
	}
	comb := []string{"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"}
	var ans []string
	solve(0, digits, comb, &ans, "")
	return ans
}
func zeroFilledSubarray(nums []int) int64 {
	result := 0
	isValid := false
	startIndex := -1
	endIndex := -1
	for i := 0; i < len(nums); i++ {
		if nums[i] == 0 {
			if isValid {
				endIndex = i
			} else {
				isValid = true
				startIndex = i
				endIndex = i
			}
			if i == len(nums)-1 {
				n := endIndex - startIndex + 1
				result += n + (n-1)*n/2
			}
		} else if nums[i] != 0 && isValid {
			n := endIndex - startIndex + 1
			result += n + (n-1)*n/2
			isValid = false
		}
	}
	return int64(result)
}

func repeatedCharacter(s string) byte {
	vector := 0

	for _, c := range s {
		if vector&(1<<(c-'a')) > 0 {
			return byte(c)
		}
		vector |= 1 << (c - 'a')
	}
	return ' '
}

func equalPairs(grid [][]int) int {
	var result int = 0
	hashedRow := make(map[string]int)
	for i := 0; i < len(grid); i++ {
		var newRow string = ""
		for j := 0; j < len(grid[i]); j++ {
			newRow += string(grid[i][j] * 10)
		}
		hashedRow[newRow]++
	}
	for i := 0; i < len(grid[0]); i++ {
		var newRow string = ""
		for j := 0; j < len(grid); j++ {
			newRow += string(grid[j][i] * 10)
		}
		if hashedRow[newRow] > 0 {
			result += hashedRow[newRow]
		}
	}
	return result
}

func getHappyString(n int, k int) string {
	stringStack := []string{""}
	index := 0
	for len(stringStack) > 0 {
		current := stringStack[len(stringStack)-1]
		stringStack = stringStack[:len(stringStack)-1]
		if len(current) == n {
			index++
			if index == k {
				return current
			}
			continue
		}
		for i := 'c'; i >= 'a'; i-- {
			if len(current) == 0 || current[len(current)-1] != uint8(i) {
				stringStack = append(stringStack, current+string(i))
			}
		}
	}
	return ""
}

func maximumsSplicedArray(nums1 []int, nums2 []int) int {
	sum1 := 0
	sum2 := 0
	maxProfit := 0
	maxProfit2 := 0
	currentSum := 0
	currentSum2 := 0
	for i := 0; i < len(nums1); i++ {
		sum1 += nums1[i]
		sum2 += nums2[i]
		temp := nums2[i] - nums1[i]
		temp2 := nums1[i] - nums2[i]
		currentSum = int(math.Max(float64(currentSum)+float64(temp), float64(temp)))
		maxProfit = int(math.Max(float64(maxProfit), float64(currentSum)))
		currentSum2 = int(math.Max(float64(currentSum2)+float64(temp2), float64(temp2)))
		maxProfit2 = int(math.Max(float64(maxProfit2), float64(currentSum2)))
	}
	return int(math.Max(math.Max(float64(sum1+maxProfit), float64(sum1)), math.Max(float64(sum2+maxProfit2), float64(sum2))))
}

func poorPigs(buckets int, minutesToDie int, minutesToTest int) int {
	result := 0
	for math.Pow(float64((minutesToTest/minutesToDie)+1), float64(result)) < float64(buckets) {
		result++
	}
	return result
}

func checkAround(i, j int, matrix [][]int, visited [][]bool) int {
	if i >= 0 && i < len(matrix) && j >= 0 && j < len(matrix[0]) && !visited[i][j] && matrix[i][j] != 0 {
		visited[i][j] = true
	} else {
		return 0
	}
	return int(1 + checkAround(i+1, j, matrix, visited) + checkAround(i, j+1, matrix, visited) + checkAround(i-1, j, matrix, visited) + checkAround(i, j-1, matrix, visited))
}

func countServers(grid [][]int) int {
	var localCount int = 0
	var visited [][]bool = make([][]bool, len(grid))
	for index, _ := range grid {
		visited[index] = make([]bool, len(grid[index]))
	}
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[i]); j++ {
			if grid[i][j] == 1 && !visited[i][j] {
				temp := int(checkAround(i, j, grid, visited))
				if temp != 1 {
					localCount += temp
				}
			}
		}
	}
	return localCount
}
func findDifferentBinaryString(nums []string) string {
	result := ""
	for i := 0; i < len(nums); i++ {
		if nums[i][i] == '0' {
			result += "1"
		} else {
			result += "0"
		}
	}
	return result
}

func sumOfTheDigitsOfHarshadNumber(x int) int {
	sumOfDigits := 0
	k := x
	for x > 0 {
		sumOfDigits += x % 10
		x /= 10
	}
	if k%sumOfDigits == 0 {
		return sumOfDigits
	}
	return -1
}

func numMovesStones(a int, b int, c int) []int {
	var minMove, maxMove int
	high := int(math.Max(float64(c), math.Max(float64(a), float64(b))))
	low := int(math.Min(float64(c), math.Min(float64(a), float64(b))))
	med := a + b + c - low - high
	if med-low > 1 {
		minMove++
	}
	if high-med > 1 {
		minMove++
	}
	maxMove = high - med - 1 + med - low - 1
	return []int{minMove, maxMove}
}

func modifyString(s string) string {
	str := strings.Split(s, "")
	chars := []string{"a", "b", "c"}

	for i := range str {
		if str[i] != "?" {
			continue
		}

		for _, c := range chars {
			if (i == 0 || str[i-1] != c) && (i == len(s)-1 || str[i+1] != c) {
				str[i] = c
				break
			}
		}
	}

	return strings.Join(str[:], "")
}

func mostWordsFound(sentences []string) int {
	maxSpaceCount := 0
	for i := 0; i < len(sentences); i++ {
		spaceCount := 0
		for j := 0; j < len(sentences[i]); j++ {
			if sentences[i][j] == ' ' {
				spaceCount++
			}
		}
		if spaceCount > maxSpaceCount {
			maxSpaceCount = spaceCount
		}
	}
	return maxSpaceCount + 1
}

/*
  - // This is the MountainArray's API interface.
  - // You should not implement it, or speculate about its implementation
  - type MountainArray struct {
  - }
    *
  - func (this *MountainArray) get(index int) int {}
  - func (this *MountainArray) length() int {}

import(
"fmt"
)

	func findInMountainArray(target int, mountainArr *MountainArray) int {
		n := mountainArr.length()
		if mountainArr.get(0) > target && mountainArr.get(n-1) > target{
			return -1
		}
		left := 0
		right := n - 1
		var peak int
		for left < right{
			mid := left + (right - left) / 2
			if mountainArr.get(mid) < mountainArr.get(mid + 1){
				left = mid + 1
			} else {
				right = mid
			}
		}
		fmt.Println(left)
		peak = left
		peakValue := mountainArr.get(peak)
		if target > peakValue{
			return  -1
		}

		if peakValue == target{
			return peak
		} else {
			result := -1
			if mountainArr.get(0) <= target {
				left = 0
				right = peak
				for left <= right{
					fmt.Println()
					fmt.Println(left)
					fmt.Println(right)
					med := left + (right-left)/2
					temp := mountainArr.get(med)
					if temp == target{
						result = med
						break
					} else if temp < target{
						left = med+1
					} else {
						right = med-1
					}
				}
				if result != -1 {
					return result
				}
			}
			if mountainArr.get(n-1) <= target {
				left = peak
				right = n-1
				for left <= right{
					med := left + (right-left)/2
					fmt.Println()
					fmt.Println(left)
					fmt.Println(right)
					temp := mountainArr.get(med)
					if temp == target{
						result = med
						break
					} else if temp > target{
						left = med+1
					} else {
						right = med-1
					}
				}
			}
			return result
		}
	}
*/

//// SeatManager seat construction
//type SeatManager struct {
//	seats []bool
//	min   int
//}
//
//func SeatConstructor(n int) SeatManager {
//	return SeatManager{make([]bool, n+1), 1}
//}
//
//func (this *SeatManager) Reserve() int {
//	for this.seats[this.min] {
//		this.min++
//	}
//	this.seats[this.min] = true
//	return this.min
//}
//
//func (this *SeatManager) Unreserve(seatNumber int) {
//	this.seats[seatNumber] = false
//	if seatNumber < this.min {
//		this.min = seatNumber
//	}
//}

func numberOfSubarrays(nums []int, k int) int {
	prefixOddCount := make([]int, len(nums)+1)
	temp := 0
	for i := 0; i < len(nums)+1; i++ {
		prefixOddCount[i] = temp
		if i < len(nums) && nums[i]%2 == 1 {
			temp++
		}
	}
	result := 0
	left := 0
	right := 1
	for right < len(prefixOddCount) {
		if prefixOddCount[right]-prefixOddCount[left] == k {
			i := left + 1
			j := right + 1
			for prefixOddCount[i] == prefixOddCount[left] && i < right {
				i++
			}
			for j < len(prefixOddCount) && prefixOddCount[j] == prefixOddCount[right] {
				j++
			}
			result += (i - left) * (j - right)
			left = i
			right = j

		} else if prefixOddCount[right]-prefixOddCount[left] < k {
			right++
		}
	}
	return result
}

func secondHighest(s string) int {
	largestTwo := []int{-1, -1}
	for i := 0; i < len(s); i++ {
		num, err := strconv.Atoi(string(s[i]))
		if err != nil {
			continue
		}
		if num > largestTwo[0] {
			largestTwo[1] = largestTwo[0]
			largestTwo[0] = num
		} else if num < largestTwo[0] {
			if num > largestTwo[1] {
				largestTwo[1] = num
			}
		}
	}
	return largestTwo[1]
}

func canArrange(arr []int, k int) bool {
	storage := make([]int, k)
	for i := 0; i < len(arr); i++ {
		storage[((arr[i]%k)+k)%k]++
	}
	if storage[0]%2 == 1 {
		return false
	}
	for i := 1; i < len(storage); i++ {
		if storage[i] != storage[k-i] {
			return false
		}
	}
	return true
}

func countSubarrays(nums []int, k int) int64 {
	maxEle := 0
	for i := range nums {
		maxEle = max(maxEle, nums[i])
	}
	var res int64 = 0
	count := 0
	left := 0
	for right := range nums {
		if nums[right] == maxEle {
			count++
		}
		for left <= right && count >= k {
			res += int64(len(nums) - right)
			if nums[left] == maxEle {
				count--
			}
			left++
		}
	}
	return res
}

func distMoney(money int, children int) int {
	switch {
	case children == 0 || money < children:
		return -1
	case children == 1 && money == 4:
		return -1
	case children == 1 && money == 8:
		return 1
	case children == 1:
		return 0
	}

	dpPrevious := distMoney(money-8, children-1)
	switch {
	case dpPrevious == -1 && children >= 2:
		return 0
	case dpPrevious == -1:
		return -1
	default:
		return 1 + dpPrevious
	}
}

func waysToBuyPensPencils(total int, cost1 int, cost2 int) int64 {
	result := int64(0)
	pencil := 0
	for total-pencil*cost1 >= 0 {
		result += int64((total-pencil*cost1)/cost2 + 1)
		pencil++
	}
	return result
}

func numOfSubarrays(arr []int) int {
	modulo := int64(math.Pow(10, 9) + 7)
	result := 0
	prefixSum := make([]int, len(arr)+1)
	prefixSum[0] = 0
	oddCount := 0
	evenCount := 1
	for i := 1; i < len(prefixSum); i++ {
		prefixSum[i] = prefixSum[i-1] + arr[i-1]
		if prefixSum[i]%2 == 0 {
			evenCount++
			result += oddCount
		} else {
			oddCount++
			result += evenCount
		}
	}

	return int(int64(result) % modulo)
}

func maxAbsoluteSum(nums []int) int {
	minPrefixSum := 0
	maxPrefixSum := 0
	prefixSum := 0
	for _, num := range nums {
		prefixSum += num
		minPrefixSum = min(minPrefixSum, prefixSum)
		maxPrefixSum = max(maxPrefixSum, prefixSum)
	}

	return maxPrefixSum - minPrefixSum
}

func search(nums []int, target int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := left + (right-left)/2
		if nums[mid] == target {
			return mid
		} else {
			if nums[mid] > target {
				right = mid - 1
			} else {
				left = mid + 1
			}
		}
	}
	return -1
}

func hIndex(citations []int) int {
	left, right := 0, len(citations)-1
	for left <= right {
		mid := left + (right-left)/2
		if citations[mid] >= len(citations)-mid {
			right = mid - 1
		} else {
			left = mid + 1
		}
	}
	return len(citations) - left
}

func canCompleteCircuit(gas []int, cost []int) int {
	totalGas, totalCost, startingPoint, currentGas := 0, 0, 0, 0
	for i := 0; i < len(gas); i++ {
		totalGas += gas[i]
		totalCost += cost[i]
		currentGas += gas[i] - cost[i]
		if currentGas < 0 {
			startingPoint = i + 1
			currentGas = 0
		}
	}
	if totalGas < totalCost {
		return -1
	} else {
		return startingPoint
	}
}

func transformArray(nums []int) []int {
	result := make([]int, len(nums))
	indexFromLast := len(nums) - 1
	indexFromFirst := 0
	for i := 0; i < len(nums) && indexFromFirst <= indexFromLast; i++ {
		if nums[i]%2 == 1 {
			result[indexFromLast] = 1
			indexFromLast--
		} else {
			result[indexFromFirst] = 0
			indexFromFirst++
		}
	}
	return result
}
func countArrays(original []int, bounds [][]int) int {
	leftBound, rightBound := bounds[0][0], bounds[0][1]
	for i := 1; i < len(bounds); i++ {
		diff := original[i] - original[i-1]
		leftBound = leftBound + diff
		rightBound = rightBound + diff
		var r, z = mutualInterval(leftBound, rightBound, bounds[i][0], bounds[i][1])
		if r == 0 && z == 0 {
			return 0
		}
		leftBound = r
		rightBound = z
	}
	return rightBound - leftBound + 1
}
func mutualInterval(a, b, c, d int) (int, int) {
	// Check if there is an overlap
	if b >= c && d >= a {
		// Compute the mutual interval
		start := max(a, c)
		end := min(b, d)
		return start, end
	}

	// No overlap
	return 0, 0
}

// Helper function to compute the minimum of two integers
func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

// Helper function to compute the maximum of two integers
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}

func largestInteger(nums []int, k int) int {
	if len(nums) == 1 {
		return nums[0]
	}
	globalCounter := make(map[int]int)
	localCounter := make(map[int]int)
	for i := 0; i < k; i++ {
		localCounter[nums[i]]++
	}
	for key, _ := range localCounter {
		globalCounter[key]++
	}
	result := -1
	for i := k; i < len(nums); i++ {
		localCounter[nums[i]]++
		localCounter[nums[i-k]]--
		if localCounter[nums[i-k]] == 0 {
			delete(localCounter, nums[i-k])
		}
		for key, _ := range localCounter {
			globalCounter[key]++
		}
	}
	for key, value := range globalCounter {
		if value == 1 && key > result {
			result = key
		}
	}
	return result
}

func mergeArrays(nums1 [][]int, nums2 [][]int) [][]int {
	m, n, firstPointer, lastPointer := len(nums1), len(nums2), 0, 0
	result := make([][]int, 0)
	for firstPointer < m || lastPointer < n {
		temp := make([]int, 2)
		var index int
		if firstPointer >= m {
			index = nums2[lastPointer][0]
		} else if lastPointer >= n {
			index = nums1[firstPointer][0]
		} else {
			index = min(nums1[firstPointer][0], nums2[lastPointer][0])
		}

		temp[0] = index
		if firstPointer < m && nums1[firstPointer][0] == index {
			temp[1] += nums1[firstPointer][1]
			firstPointer++
		}
		if lastPointer < n && nums2[lastPointer][0] == index {
			temp[1] += nums2[lastPointer][1]
			lastPointer++
		}
		result = append(result, temp)
	}
	return result
}

func pivotArray(nums []int, pivot int) []int {
	less, equal, bigger := make([]int, 0), make([]int, 0), make([]int, 0)
	for i := 0; i < len(nums); i++ {
		if nums[i] < pivot {
			less = append(less, nums[i])
		} else if nums[i] > pivot {
			bigger = append(bigger, nums[i])
		} else {
			equal = append(equal, nums[i])
		}
	}
	for _, value := range equal {
		less = append(less, value)
	}
	for _, value := range bigger {
		less = append(less, value)
	}
	return less
}

func validSquare(p1 []int, p2 []int, p3 []int, p4 []int) bool {
	var diff12, diff13, diff14, diff23, diff24, diff34 int64
	coll := [][]int{p1, p2, p3, p4}
	for i := 0; i < len(coll)-1; i++ {
		for j := i + 1; j < len(coll[i]); j++ {
			if coll[i][0] == coll[j][0] && coll[i][1] == coll[j][1] {
				return false
			}
		}
	}
	dict := make(map[int64]int)
	diff12 = int64(math.Abs(float64(p1[0]-p2[0]))*math.Abs(float64(p1[0]-p2[0])) + math.Abs(float64(p1[1]-p2[1]))*math.Abs(float64(p1[1]-p2[1])))
	dict[diff12]++

	diff14 = int64(math.Abs(float64(p1[0]-p4[0]))*math.Abs(float64(p1[0]-p4[0])) + math.Abs(float64(p1[1]-p4[1]))*math.Abs(float64(p1[1]-p4[1])))
	dict[diff14]++

	diff23 = int64(math.Abs(float64(p2[0]-p3[0]))*math.Abs(float64(p2[0]-p3[0])) + math.Abs(float64(p2[1]-p3[1]))*math.Abs(float64(p2[1]-p3[1])))
	dict[diff23]++

	diff34 = int64(math.Abs(float64(p4[0]-p3[0]))*math.Abs(float64(p4[0]-p3[0])) + math.Abs(float64(p4[1]-p3[1]))*math.Abs(float64(p4[1]-p3[1])))
	dict[diff34]++

	diff13 = int64(math.Abs(float64(p1[0]-p3[0]))*math.Abs(float64(p1[0]-p3[0])) + math.Abs(float64(p1[1]-p3[1]))*math.Abs(float64(p1[1]-p3[1])))
	dict[diff13]++

	diff24 = int64(math.Abs(float64(p2[0]-p4[0]))*math.Abs(float64(p2[0]-p4[0])) + math.Abs(float64(p2[1]-p4[1]))*math.Abs(float64(p2[1]-p4[1])))
	dict[diff24]++
	if len(dict) != 2 {
		return false
	} else {
		var equalLength int64 = math.MaxInt64
		var maxLength int64 = math.MinInt64
		for key, _ := range dict {
			if key < equalLength {
				equalLength = key
			}
			if key > maxLength {
				maxLength = key
			}
		}
		if dict[equalLength] == 4 && dict[maxLength] == 2 {
			return true
		} else {
			return false
		}
	}
}

func coloredCells(n int) int64 {
	return int64(2*n*n - 2*n + 1)
}

func findMissingAndRepeatedValues(grid [][]int) []int {
	n := len(grid)
	result := make([]int, 2)
	sum := 0
	squareSum := int64(0)
	idealSum := (n * n) * (n*n + 1) / 2
	idealSquareSum := int64((n*n)*(n*n+1)*(2*n*n+1)) / 6
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[i]); j++ {
			sum += grid[i][j]
			squareSum += int64(grid[i][j] * grid[i][j])
		}
	}
	diff := int64(sum - idealSum)
	diffSquareSum := squareSum - idealSquareSum
	result[0] = int(diffSquareSum/diff+diff) / 2
	result[1] = result[0] - int(diff)
	return result
}
func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func minOperations(boxes string) []int {
	result := make([]int, len(boxes))
	for i := 0; i < len(boxes); i++ {
		for j := 0; j < len(boxes); j++ {
			if j != i && boxes[j] == '1' {
				result[i] += abs(j - i)
			}
		}
	}
	return result
}

func scoreOfString(s string) int {
	sum := 0
	for i := 1; i < len(s); i++ {
		sum += abs(int(s[i]) - int(s[i-1]))
	}
	return sum
}

func findArray(pref []int) []int {
	result := make([]int, len(pref))
	result[0] = pref[0]
	for i := 1; i < len(pref); i++ {
		result[i] = result[i-1] ^ pref[i]
	}
	return result
}

//func maximalRectangle(matrix [][]byte) int {
//	arr := matrix[0]
//	for i := 0; i < len(matrix); i++ {
//
//	}
//}

func largestRectangleArea(heights []int) int {
	var stack []int
	maxArea := 0
	n := len(heights)

	for i := 0; i <= n; i++ {
		currentHeight := 0
		if i < n {
			currentHeight = heights[i]
		}

		for len(stack) > 0 && heights[stack[len(stack)-1]] > currentHeight {
			height := heights[stack[len(stack)-1]]
			stack = stack[:len(stack)-1]
			width := i
			if len(stack) > 0 {
				width = i - stack[len(stack)-1] - 1
			}
			maxArea = max(maxArea, height*width)
		}

		stack = append(stack, i)
	}

	return maxArea
}

func maxScoreSightseeingPair(values []int) int {
	maxScore := 0
	bestLeft := values[0]
	for j := 1; j < len(values); j++ {
		maxScore = max(maxScore, bestLeft+values[j]-j)
		bestLeft = max(bestLeft, values[j]+j)
	}
	return maxScore
}

func climbStairs(n int) int {
	dp := make([]int, n+1)
	dp[0] = 1
	dp[1] = 1
	for i := 2; i <= n; i++ {
		dp[i] = dp[i-1] + dp[i-2]
	}
	return dp[n]
}

func main() {
	//p1 := []int{1, 0}
	//p2 := []int{0, 1}
	//p3 := []int{-1, 0}
	//p4 := []int{0, -1}

	fmt.Print(largestRectangleArea([]int{0, 9}))
}

/* Randomized Set
type RandomizedSet struct {
	arr  []int
	size int
	set  map[int]int
}

func Constructor() RandomizedSet {
	arr := make([]int, 0)
	size := 0
	set := make(map[int]int)
	return RandomizedSet{
		arr:  arr,
		size: size,
		set:  set,
	}
}

func (this *RandomizedSet) Insert(val int) bool {
	_, ok := this.set[val]
	if ok {
		return false
	}

	this.arr = append(this.arr, val)
	this.set[val] = this.size
	this.size++
	return true
}

func (this *RandomizedSet) Remove(val int) bool {
	index, ok := this.set[val]
	if !ok {
		return false
	}

	// Swap the element to remove with the last element
	lastElement := this.arr[this.size-1]
	this.arr[index] = lastElement
	this.set[lastElement] = index

	// Remove the last element
	this.arr = this.arr[:this.size-1]
	delete(this.set, val)
	this.size--
	return true
}

func (this *RandomizedSet) GetRandom() int {
	index := rand.Intn(this.size)
	return this.arr[index]
}

*/
