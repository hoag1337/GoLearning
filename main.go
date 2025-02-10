package main

import (
	"fmt"
	"slices"
	"sort"
	"strings"
	"unicode"
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

func Constructor() NumberContainers {
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

type Stack struct {
	data []interface{}
	top  int
}

func (s *Stack) Push(element interface{}) {
	s.top++
	s.data = append(s.data, element)
}

func (s *Stack) Pop() interface{} {
	if len(s.data) > 0 {
		s.top--
		last := s.data[s.top]
		s.data = s.data[:s.top]

		return last
	}

	return nil
}

func (s Stack) DataToString() string {
	var strData []string
	for _, v := range s.data {
		strData = append(strData, fmt.Sprintf("%v", v))
	}
	return strings.Join(strData, ", ")
}
func clearDigits(s string) string {
	notDigitStack := Stack{}
	for i := 0; i < len(s); i++ {
		if unicode.IsDigit(rune(s[i])) {
			notDigitStack.Pop()
		} else {
			notDigitStack.Push(s[i])
		}
	}
	return notDigitStack.DataToString()
}
func main() {
	fmt.Println(clearDigits("abc12"))
}
